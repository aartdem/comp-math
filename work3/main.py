import os
import struct
import time
import warnings
from enum import Enum
import numpy as np
from PIL import Image

MY_FORMAT_HEADER = b'CBMP'
MY_FORMAT_HEADER_SIZE = 16


class Method(Enum):
    NUMPY = 1
    POWER_SIMPLE = 2
    SMART = 3


def compress_bmp(input_file: str, output_file: str, ratio: float, method: Method):
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
                compressed_channel = compress_channel_power_simple(channel, k, 100000)
            case Method.SMART:
                raise ValueError('Is not implemented')

        compressed_image_data += compressed_channel

    # write data to output_file
    with open(output_file, 'wb') as f:
        header_data = MY_FORMAT_HEADER + struct.pack('<i', n) + struct.pack('<i', m) + struct.pack('<i', k)
        f.write(header_data)
        f.write(compressed_image_data)

    if ratio * os.path.getsize(output_file) > os.path.getsize(input_file):
        warnings.warn(f'The size of the compressed file is larger than the (size of the original) / {ratio}',
                      RuntimeWarning)


def compress_channel_numpy(channel: np.ndarray, k: int):
    u, sigma, vt = np.linalg.svd(channel, full_matrices=False)
    data = np.concatenate((u[:, :k].ravel(), sigma[:k], vt[:k, :].ravel()))
    return data.astype(np.float32).tobytes()


def compress_channel_power_simple(a: np.ndarray, k: int, duration: int):
    eps = 1e-10
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
            if np.allclose(v_new, v, eps):
                break
            v = v_new

        eigenvalue = np.dot(np.dot(ata, v), v.T)
        vt[i, :] = v
        u[:, i] = np.dot(a, v) / eigenvalue
        sigma[i] = eigenvalue

        a = a - eigenvalue * np.outer(u[:, i], v)

    data = np.concatenate((u.ravel(), sigma, vt.ravel()))
    return data.astype(np.float32).tobytes()


def unpack_channel(byte_data, n, m, k):
    split_data = [byte_data[i:i + 4] for i in range(0, len(byte_data), 4)]
    map_obj = map(lambda x: struct.unpack('<f', x), split_data)
    matrix_data = np.array(list(map_obj))
    u = matrix_data[: n * k].reshape(n, k)
    sigma = matrix_data[n * k: n * k + k].ravel()
    vt = matrix_data[n * k + k:].reshape(k, m)
    return np.dot(np.dot(u, np.diag(sigma)), vt)


def restore_image(input_file, result_image_name):
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


path_to_output = 'output.cbmp'
path_to_result_image = 'result.bmp'
# path_to_image = 'src-images/XING_B24.BMP'
path_to_image = 'src-images/nature.bmp'
compress_ratio = 2

if __name__ == '__main__':
    compress_bmp(path_to_image, path_to_output, compress_ratio, Method.POWER_SIMPLE)
    restore_image(path_to_output, path_to_result_image)
    # os.remove(path_to_output)
