import argparse
import os
import struct
import time
import warnings
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

MY_FORMAT_HEADER = b'CBMP'
MY_FORMAT_HEADER_SIZE = 16
EPS = 1e-10


class Method(Enum):
    NUMPY = 1
    POWER_SIMPLE = 2
    BLOCK_POWER = 3


def compress_bmp(input_file: str, output_file: str, ratio: float, method: Method) -> None:
    # read original image
    img = Image.open(input_file)
    n, m = img.height, img.width
    k = np.floor(n * m / (4 * ratio * (n + m + 1))).astype(np.int32)

    # calculate data of compressed image
    img_arrays = np.asarray(img)
    compressed_image_data = bytes()
    for i in range(3):
        channel = img_arrays[..., i]
        compressed_channel = bytes()
        match method:
            case Method.NUMPY:
                compressed_channel = compress_channel_numpy(channel, k)
            case Method.POWER_SIMPLE:
                compressed_channel = compress_channel_power_simple(channel, k, 60000)
            case Method.BLOCK_POWER:
                compressed_channel = compress_channel_block_power(channel, k, 60000)

        compressed_image_data += compressed_channel

    # write data to output_file
    with open(output_file, 'wb') as f:
        header_data = MY_FORMAT_HEADER + struct.pack('<i', n) + struct.pack('<i', m) + struct.pack('<i', k)
        f.write(header_data)
        f.write(compressed_image_data)

    # check output_file size
    if ratio * os.path.getsize(output_file) > os.path.getsize(input_file):
        warnings.warn(f'The size of the compressed file is larger than the (size of the original) / {ratio}',
                      RuntimeWarning)


def compress_channel_numpy(channel: np.ndarray, k: int) -> bytes:
    u, sigma, vt = np.linalg.svd(channel, full_matrices=False)
    data = np.concatenate((u[:, :k].ravel(), sigma[:k], vt[:k, :].ravel()))
    return data.astype(np.float32).tobytes()


# Simple power method
def compress_channel_power_simple(a: np.ndarray, k: int, duration: int) -> bytes:
    np.random.seed(0)
    n, m = a.shape
    v = np.random.rand(m)
    v /= np.linalg.norm(v)
    u = np.zeros((n, k))
    sigma = np.zeros(k)
    vt = np.zeros((k, m))

    time_bound = time.time() * 1000 + duration
    for i in range(k):
        print(i)
        ata = np.dot(a.T, a)
        while time.time() * 1000 < time_bound:
            v_new = np.dot(ata, v)
            v_new /= np.linalg.norm(v_new)
            if np.allclose(v_new, v, EPS):
                break
            v = v_new

        eigenvalue = np.dot(np.dot(ata, v), v.T)
        vt[i, :] = v
        u[:, i] = np.dot(a, v) / eigenvalue
        sigma[i] = eigenvalue

        a = a - eigenvalue * np.outer(u[:, i], v)

    data = np.concatenate((u.ravel(), sigma, vt.ravel()))
    return data.astype(np.float32).tobytes()


# Block power method taken from [https://sciendo.com/article/10.1515/auom-2015-0024?content-tab=abstract]
def compress_channel_block_power(a: np.ndarray, k: int, duration: int) -> bytes:
    np.random.seed(0)
    n, m = a.shape
    u = np.zeros((n, k))
    sigma = np.zeros(k)
    v = np.zeros((m, k))

    counter = 0
    time_bound = time.time() * 1000 + duration
    while time.time() * 1000 < time_bound:
        q, _ = np.linalg.qr(np.dot(a, v))
        u = q[:, :k]
        q, r = np.linalg.qr(np.dot(a.T, u))
        v = q[:, :k]
        sigma = np.diag(r[:k, :k])
        counter += 1
        print(counter)
        if counter % 5 == 0 and np.allclose(np.dot(a, v), np.dot(u, r[:k, :k]), EPS):
            break

    data = np.concatenate((u.ravel(), sigma, v.T.ravel()))
    return data.astype(np.float32).tobytes()


def unpack_channel(byte_data, n, m, k) -> np.ndarray:
    split_data = [byte_data[i:i + 4] for i in range(0, len(byte_data), 4)]
    map_obj = map(lambda x: struct.unpack('<f', x), split_data)
    matrix_data = np.array(list(map_obj))
    u = matrix_data[: n * k].reshape(n, k)
    sigma = matrix_data[n * k: n * k + k].ravel()
    vt = matrix_data[n * k + k:].reshape(k, m)
    return np.dot(np.dot(u, np.diag(sigma)), vt)


def restore_image(input_file, result_image_name) -> None:
    with open(input_file, 'rb') as f:
        header_data = f.read(MY_FORMAT_HEADER_SIZE)
        if header_data[:4] != MY_FORMAT_HEADER:
            raise ValueError(f'Incorrect format of {input_file}')
        n = struct.unpack('<i', header_data[4:8])[0]
        m = struct.unpack('<i', header_data[8:12])[0]
        k = struct.unpack('<i', header_data[12:16])[0]
        arrays = [unpack_channel(f.read(4 * k * (n + m + 1)), n, m, k) for _ in range(3)]
        image_matrix = np.stack(arrays, axis=2).clip(0, 255).astype(np.uint8)
        result_image = Image.fromarray(image_matrix)
        result_image.save(result_image_name)


path_to_result_image = 'result.bmp'
# path_to_image = 'src-images/XING_B24.BMP'
path_to_image = 'src-images/nature.bmp'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to image')
    parser.add_argument('-r', '--ratio', type=float, required=True,
                        help='The ratio of the size of the source file and the intermediate representation')
    parser.add_argument('-m', '--method', type=str, required=True, choices=['numpy', 'power_simple', 'block_power'],
                        help='The method used to calculate SVD')
    parser.add_argument('-s', '--save', action='store_true',help='Enable saving an intermediate representation')

    args = parser.parse_args()
    path_to_result_image = f'{Path(args.path).stem}-{args.method}.bmp'
    temp_output = f'{Path(args.path).stem}-compressed.cbmp'

    # Main steps
    match args.method:
        case 'numpy':
            compress_bmp(path_to_image, temp_output, args.ratio, Method.NUMPY)
        case 'power_simple':
            compress_bmp(path_to_image, temp_output, args.ratio, Method.POWER_SIMPLE)
        case 'block_power':
            compress_bmp(path_to_image, temp_output, args.ratio, Method.BLOCK_POWER)
    restore_image(temp_output, path_to_result_image)
    if not args.save:
        os.remove(temp_output)
