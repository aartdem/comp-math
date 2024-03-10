#include <iostream>
#include <omp.h>
#include <vector>
#include <functional>
#include <cmath>
#include <iomanip>
#include <csignal>
#include <fcntl.h>

class NetSolver {
public:
    NetSolver(int n, int bsize, double eps, const std::function<double(double, double)> &g_fun_,
              const std::function<double(double, double)> &f_fun_) :
            expected_fun(g_fun_), N(n), h(1.0 / (n + 1)), BSIZE(bsize), NB(n / bsize + (n % bsize != 0)), EPS(eps) {
        u_arr.resize(N + 2, std::vector<double>(N + 2, 0));
        f_arr.resize(N + 2, std::vector<double>(N + 2, 0));
        double mean = 0;
        for (int i = 0; i <= N + 1; i++) {
            u_arr[i][0] = g_fun_(i * h, 0);
            u_arr[i][N + 1] = g_fun_(i * h, (N + 1) * h);
            u_arr[0][i] = g_fun_(0, i * h);
            u_arr[N + 1][i] = g_fun_((N + 1) * h, i * h);
            mean += (u_arr[i][0] + u_arr[i][N + 1] + u_arr[0][i] + u_arr[N + 1][i]);
        }
        mean = (mean - u_arr[0][0] - u_arr[0][N + 1] - u_arr[N + 1][0] - u_arr[N + 1][N + 1]) / ((N + 1) * 4);
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                f_arr[i][j] = f_fun_(i * h, j * h);
//                u_arr[i][j] = mean;
            }
        }
    }

    void calculate_approximation() {
        std::vector<double> dm(NB, 0);
        double dmax;
        do {
            dmax = 0;
            std::fill(dm.begin(), dm.end(), 0);
            for (int nx = 0; nx < NB; nx++) {
#pragma omp parallel for shared(nx, dm, u_arr) default(none)
                for (int i = 0; i <= nx; i++) {
                    int j = nx - i;
                    double d = block_processing(i, j);
                    dm[i] = std::max(dm[i], d);
                }
            }
            for (int nx = NB - 2; nx >= 0; nx--) {
#pragma omp parallel for shared(nx, dm, u_arr) default(none)
                for (int i = NB - nx - 1; i < NB; i++) {
                    int j = 2 * (NB - 1) - nx - i;
                    double d = block_processing(i, j);
                    dm[i] = std::max(dm[i], d);
                }
            }
            for (int i = 0; i < NB; i++) {
                dmax = std::max(dmax, dm[i]);
            }
        } while (dmax > EPS);
    }

    double calculate_max_error() {
        double max_error = 0;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                max_error = std::max(max_error, std::abs(u_arr[i][j] - expected_fun(i * h, j * h)));
            }
        }
        return max_error;
    }

    double calculate_mean_absolute_error() {
        double sum_error = 0;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                sum_error += std::abs(u_arr[i][j] - expected_fun(i * h, j * h));
            }
        }
        return sum_error / (N * N);
    }

    double calculate_approximate_error() {
        double sum_error = 0;
        int zero_count = 0;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                double expected = expected_fun(i * h, j * h);
                // skip point if expected function value is very close to 0
                if (std::abs(expected) > ZERO) {
                    sum_error += std::abs(u_arr[i][j] - expected) / std::abs(expected);
                } else {
                    zero_count++;
                }
            }
        }
        return sum_error / (N * N - zero_count);
    }

private:
    double block_processing(int block_i, int block_j) {
        double dmax = 0;
        for (int i = 1 + block_i * BSIZE; i <= std::min((block_i + 1) * BSIZE, N); i++) {
            for (int j = 1 + block_j * BSIZE; j <= std::min((block_j + 1) * BSIZE, N); j++) {
                double temp = u_arr[i][j];
                u_arr[i][j] = 0.25 * std::abs(u_arr[i - 1][j] + u_arr[i + 1][j] +
                                              u_arr[i][j - 1] + u_arr[i][j + 1] - h * h * f_arr[i][j]);
                dmax = std::max(dmax, std::abs(temp - u_arr[i][j]));
            }
        }
        return dmax;
    }

    const int N, NB, BSIZE;
    const double ZERO = 1e-15, EPS, h;
    std::vector<std::vector<double>> u_arr, f_arr;
    const std::function<double(double, double)> expected_fun;
};

//auto test_fun = [](double x, double y) {
//    return 0.5 * x * x + y * y - y - 0.5 * x * y + 0.25;
//};
//auto test_fun_d = [](double x , double y) { return 3; };
//auto test_fun = [](double x, double y) { return pow(x, 6) + pow(y, 6); };
//auto test_fun_d = [](double x, double y) { return 30 * pow(x, 4) + 30 * pow(y, 4); };
//auto test_fun = [](double x, double y) { return sin(x) + sin(y); };
//auto test_fun_d = [](double x, double y) { return -sin(x) - sin(y); };
#define THREADS_NUM 4

void test_different_net_sizes() {
    omp_set_num_threads(THREADS_NUM);
    auto test_fun = [](double x, double y) {
        return 0.5 * x * x + y * y - y - 0.5 * x * y + 0.25;
    };
    auto test_fun_d = [](double x, double y) { return 3; };
    std::vector<int> test_net_sizes{10, 50, 100, 500, 1000, 5000};
    for (auto it: test_net_sizes) {
        NetSolver net(it, 16, 1e-2, test_fun, test_fun_d);

        auto start_time = omp_get_wtime();
        net.calculate_approximation();
        auto end_time = omp_get_wtime();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "-------| Net size = " << it << " |-------\n";
        std::cout << "WORK TIME: " << end_time - start_time << '\n';
        std::cout << "APPROXIMATE ERROR: " << net.calculate_approximate_error() << '\n';
    }
}

int main() {
    omp_set_num_threads(THREADS_NUM);
    auto test_fun = [](double x, double y) { return sin(x) + sin(y); };
    auto test_fun_d = [](double x, double y) { return -sin(x) - sin(y); };
    freopen("eps_results.txt", "w", stdout);
    double step = 1e-5;
    for (double eps = step; eps <= 0.1; eps += step) {
        NetSolver net(100, 16, eps, test_fun, test_fun_d);
        auto start_time = omp_get_wtime();
        net.calculate_approximation();
        auto end_time = omp_get_wtime();
        std::cout << std::fixed << std::setprecision(6) << eps << ' ' << (end_time - start_time) << ' '
                  << net.calculate_approximate_error() << '\n';
    }
}
