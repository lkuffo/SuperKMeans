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
    // Choose dataset by name. You can also pass the dataset name as the first CLI argument.
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("mxbai");

    const std::unordered_map<std::string, std::pair<size_t, size_t>> dataset_params = {
        {"mxbai", {769382, 1024}}, // pd: 128
        {"openai", {999000, 1536}}, // pd: 128
        {"arxiv", {2253000, 768}},  //
        {"sift", {1000000, 128}}, // pd: 32
        {"fmnist", {60000, 784}} // pd: 32
    };

    auto it = dataset_params.find(dataset);
    if (it == dataset_params.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, arxiv, sift, fmnist\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters =
        std::max<size_t>(1u, static_cast<size_t>(std::sqrt(static_cast<double>(n)) * 4.0));
    int n_iters = 25;
    float sampling_fraction = 1.0;
    std::string path_root = std::string(CMAKE_SOURCE_DIR);
    std::string filename = path_root + "/data_" + dataset + ".bin";
    constexpr size_t THREADS = 10;
    omp_set_num_threads(THREADS);

    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    std::vector<skmeans::skmeans_value_t<skmeans::f32>> data;
    try {
        data.resize(n * d);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what()
                  << "\n";
        return 1;
    }

    std::ifstream file(filename, std::ios::binary);
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
