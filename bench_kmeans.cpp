#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/pruners/adsampling.hpp"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

int main(int argc, char* argv[]) {
    std::cout << "Compiles!" << std::endl;

    // SKMeans
    const size_t n = 262144;
    // const size_t n = 720896;
    const size_t d = 1024;
    size_t n_clusters = 1024;
    uint32_t n_iters = 2;
    float sampling_fraction = 1.0;

    constexpr size_t THREADS = 10;
    omp_set_num_threads(THREADS);
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    std::vector<skmeans::skmeans_value_t<skmeans::f32>> data(n * d);
    std::ifstream file(std::string{CMAKE_SOURCE_DIR} + "/data_mxbai.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    auto kmeans_state = skmeans::SuperKMeans<skmeans::f32, skmeans::l2>(
        n_clusters, d, n_iters, sampling_fraction, true, THREADS
    );
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("SKMeans", [&]() {
        auto centroids = kmeans_state.Train(data.data(), n);
    });
}
