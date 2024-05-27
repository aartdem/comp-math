import numpy as np
from scipy.integrate import quad

lambda1: float


def f(x: float) -> float:
    return 2 * lambda1 * np.sin(np.sqrt(lambda1) * x)


def equation_solution(x: float) -> float:
    return np.sin(np.sqrt(lambda1) * x)


def p(_: float) -> float:
    return 1


def q(_: float) -> float:
    return lambda1


# Calculates value of the j-th basic function in point x
def phi_j(j: int, x: float, xs: np.array) -> float:
    n = xs.size
    if j < 0 or j >= n:
        return 0
    elif j == 0:
        if x <= xs[1]:
            return (xs[1] - x) / (xs[1] - xs[0])
        else:
            return 0
    elif j == n - 1:
        if x >= xs[n - 1]:
            return (x - xs[n - 1]) / (xs[n - 1] - xs[n - 2])
        else:
            return 0
    else:
        if xs[j - 1] <= x <= xs[j]:
            return (x - xs[j - 1]) / (xs[j] - xs[j - 1])
        elif xs[j] <= x <= xs[j + 1]:
            return (xs[j + 1] - x) / (xs[j + 1] - xs[j])
        else:
            return 0


def calculate_matrix(xs: np.array) -> np.ndarray:
    n = xs.size
    # To store the tridiagonal matrix we can use only 3 arrays of size n
    a = np.zeros((3, n))
    # Dirichlet boundary conditions
    a[1][0] = 1
    a[1][n - 1] = 1
    for j in range(1, n - 1):
        h1 = xs[j] - xs[j - 1]
        h2 = xs[j + 1] - xs[j]
        # using formula 8.69
        a[0][j] = quad(lambda x: -p(x) + q(x) * (x - xs[j - 1]) * (xs[j] - x), xs[j - 1], xs[j])[0] / (h1 ** 2)
        # using formula 8.70
        a[2][j] = quad(lambda x: -p(x) + q(x) * (x - xs[j]) * (xs[j + 1] - x), xs[j], xs[j + 1])[0] / (h2 ** 2)
        # using formula 8.71
        t1 = quad(lambda x: p(x) + q(x) * ((x - xs[j - 1]) ** 2), xs[j - 1], xs[j])[0] / (h1 ** 2)
        t2 = quad(lambda x: p(x) + q(x) * ((xs[j + 1] - x) ** 2), xs[j], xs[j + 1])[0] / (h2 ** 2)
        a[1][j] = t1 + t2
    return a


# calculate right side of equation 8.68
def calculate_vector(xs: np.array) -> np.array:
    n = xs.size
    v = np.zeros(n)
    for j in range(1, n - 1):
        v[j] = quad(lambda x: f(x) * phi_j(j, x, xs), xs[j - 1], xs[j + 1])[0]
    return v


# https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
def thomas(m: np.ndarray, d: np.array) -> np.array:
    n = d.size
    a, b, c = (m[0], m[1], m[2])

    for j in range(1, n):
        w = a[j] / b[j - 1]
        b[j] -= w * c[j - 1]
        d[j] -= w * d[j - 1]

    y = np.zeros(n)
    y[n - 1] = d[n - 1] / b[n - 1]
    for j in range(n - 2, -1, -1):
        y[j] = (d[j] - c[j] * y[j + 1]) / b[j]

    return y


# eval value of function, works only for uniform grid
def eval_f_approx(x: float, xs: np.array, y: np.array):
    n = xs.size
    h = xs[1] - xs[0]
    i = np.floor(x / h).astype(int)
    if i == n - 1:
        return y[n - 1]
    else:
        return y[i] * phi_j(i, x, xs) + y[i + 1] * phi_j(i + 1, x, xs)


def check_error(xs: np.array, y: np.array):
    n = xs.size
    new_n = n * 10
    new_xs = np.linspace(xs[0], xs[n - 1], new_n)
    for x in new_xs:
        print(equation_solution(x), eval_f_approx(x, xs, y))


if __name__ == '__main__':
    L = 0
    R = np.pi
    N = 10  # count of knots
    lambda1 = 1

    grid = np.linspace(L, R, N)
    matr = calculate_matrix(grid)
    vec = calculate_vector(grid)
    y_result = thomas(matr, vec)
    check_error(grid, y_result)
