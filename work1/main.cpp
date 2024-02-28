#include <iostream>
#include <omp.h>
#include <vector>
#include <functional>
#include <cmath>
#include <chrono>
#include <iomanip>

class NetSolver {
public:
    NetSolver(int n, int bsize, const std::function<double(double, double)> &g_fun_,
              const std::function<double(double, double)> &f_fun_) : actual_fun(g_fun_), N(n), H(1.0 / (n + 1)),
                                                                     BSIZE(bsize),
                                                                     NB(n / bsize + (n % bsize == 0)) {
        u_arr.resize(N + 2, std::vector<double>(N + 2, 0));
        f_arr.resize(N + 2, std::vector<double>(N + 2, 0));
        for (int i = 0; i <= N + 1; i++) {
            u_arr[i][0] = g_fun_(i * H, 0);
            u_arr[i][N + 1] = g_fun_(i * H, (N + 1) * H);
            u_arr[0][i] = g_fun_(0, i * H);
            u_arr[N + 1][i] = g_fun_((N + 1) * H, i * H);
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                f_arr[i][j] = f_fun_(i * H, j * H);
            }
        }
    }

    void calculate_approximation() {
        double dmax = 0;
        std::vector<double> dm(NB, 0);
        do {
            for (int nx = 0; nx < NB; nx++) {
#pragma omp parallel for shared(nx, dm) default(none)
                for (int i = 0; i <= nx; i++) {
                    int j = nx - i;
                    double d = block_processing(i, j);
                    dm[i] = std::max(dm[i], d);
                }
            }
            for (int nx = NB - 2; nx >= 0; nx--) {
#pragma omp parallel for shared(nx, dm) default(none)
                for (int i = 1; i <= nx; i++) {
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

    double test_results() {
        double sum_error = 0;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                sum_error += fabs(u_arr[i][j] - actual_fun(i * H, j * H));
            }
        }
        return sum_error / (N * N);
    }

private:
    inline double block_processing(int block_i, int block_j) {
        double dmax = 0;
        for (int i = 1 + block_i * BSIZE; i <= std::min((block_i + 1) * BSIZE, N); i++) {
            for (int j = 1 + block_j * BSIZE; j <= std::min((block_j + 1) * BSIZE, N); j++) {
                double temp = u_arr[i][j];
                u_arr[i][j] = 0.25 * (u_arr[i - 1][j] + u_arr[i + 1][j] +
                                      u_arr[i][j - 1] + u_arr[i][j + 1] - H * H * f_arr[i][j]);
                dmax = std::max(dmax, temp - u_arr[i][j]);
            }
        }
        return dmax;
    }

    const int N, NB, BSIZE;
    const double EPS = 0.01, H;
    std::vector<std::vector<double>> u_arr, f_arr;
    const std::function<double(double, double)> actual_fun;
};

#define TREADS_NUM 4

int main() {
    omp_set_num_threads(TREADS_NUM);

    auto actual_fun = [](double x, double y) { return 2 * pow(x, 5) + 3 * pow(y, 4); };
    auto actual_fun_d = [](double x, double y) { return 40 * pow(x, 3) + 36 * pow(y, 2); };
    NetSolver net(10000, 32, actual_fun, actual_fun_d);


    auto start_time = omp_get_wtime();
    net.calculate_approximation();
    auto end_time = omp_get_wtime();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "WORK TIME: " << end_time - start_time << '\n';
    std::cout << "MEAN ERROR: " << net.test_results();
}
