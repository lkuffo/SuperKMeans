#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/utils.h>

#include "superkmeans/common.h"
#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

std::vector<float>
make_blobs(size_t n_samples, size_t n_features, size_t n_centers, unsigned int random_state = 42) {
    std::mt19937 gen(random_state);

    // Random cluster centers
    std::normal_distribution<float> center_dist(0.0f, 1.0f);
    std::vector<std::vector<float>> centers(n_centers, std::vector<float>(n_features));
    for (auto& c : centers)
        for (auto& x : c)
            x = center_dist(gen);

    // Distributions for choosing cluster and spreading points
    std::uniform_int_distribution<size_t> cluster_dist(0, n_centers - 1);
    std::normal_distribution<float> point_dist(0.0f, 1.0f);

    // Flattened result: row-major layout
    std::vector<float> data;
    data.reserve(n_samples * n_features);

    for (size_t i = 0; i < n_samples; ++i) {
        const auto& center = centers[cluster_dist(gen)];
        for (size_t j = 0; j < n_features; ++j)
            data.push_back(center[j] + point_dist(gen));
    }

    return data;
}

int main(int argc, char* argv[]) {
    std::cout << "=== SuperKMeans vs FAISS Benchmarks ===" << std::endl << std::endl;

    // Configuration
    constexpr size_t THREADS = 10;
    omp_set_num_threads(THREADS);

    std::cout << "FAISS compile options: " << faiss::get_compile_options() << std::endl;
    std::cout << "Threads: " << THREADS << std::endl << std::endl;

    // Benchmark parameters: different dataset sizes and dimensions
    struct BenchmarkConfig {
        std::string name;
        size_t n;
        size_t d;
        size_t n_clusters;
        size_t n_iters;
    };

    std::vector<BenchmarkConfig> configs = {
        {"Small_LowDim", 10000, 64, 100, 10},
        {"Medium_LowDim", 100000, 128, 1000, 15},
        {"Large_HighDim", 100000, 512, 1000, 20},
        {"VeryLarge_HighDim", 500000, 1024, 2000, 25}
    };

    for (const auto& cfg : configs) {
        std::cout << "--- Benchmark: " << cfg.name << " ---" << std::endl;
        std::cout << "n=" << cfg.n << ", d=" << cfg.d
                  << ", n_clusters=" << cfg.n_clusters
                  << ", n_iters=" << cfg.n_iters << std::endl;

        // Generate synthetic data
        std::vector<float> data = make_blobs(cfg.n, cfg.d, cfg.n_clusters);

        // Benchmark SuperKMeans
        std::cout << "SuperKMeans:" << std::endl;
        skmeans::SuperKMeansConfig config;
        config.iters = cfg.n_iters;
        config.sampling_fraction = 1.0;
        config.verbose = false;
        config.n_threads = THREADS;

        ankerl::nanobench::Bench()
            .epochs(1)
            .epochIterations(1)
            .run(cfg.name + "_SuperKMeans", [&]() {
                auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    cfg.n_clusters, cfg.d, config
                );
                auto centroids = kmeans.Train(data.data(), cfg.n);
                ankerl::nanobench::doNotOptimizeAway(centroids);
            });

        // Benchmark FAISS
        std::cout << "FAISS:" << std::endl;
        faiss::IndexFlatL2 index(cfg.d);
        faiss::ClusteringParameters cp;
        cp.niter = cfg.n_iters;
        cp.verbose = false;
        cp.nredo = 1;
        faiss::Clustering clus(cfg.d, cfg.n_clusters, cp);

        ankerl::nanobench::Bench()
            .epochs(1)
            .epochIterations(1)
            .run(cfg.name + "_FAISS", [&]() {
                // Need to reset clustering for each run
                faiss::Clustering clus_run(cfg.d, cfg.n_clusters, cp);
                faiss::IndexFlatL2 index_run(cfg.d);
                clus_run.train(cfg.n, data.data(), index_run);
                ankerl::nanobench::doNotOptimizeAway(clus_run.centroids);
            });

        std::cout << std::endl;
    }

    // Benchmark assignment performance separately
    std::cout << "--- Benchmark: Assignment Performance ---" << std::endl;
    {
        const size_t n = 100000;
        const size_t d = 256;
        const size_t n_clusters = 1024;

        std::vector<float> data = make_blobs(n, d, n_clusters);
        std::vector<float> centroids = make_blobs(n_clusters, d, n_clusters);

        ankerl::nanobench::Bench()
            .epochs(3)
            .epochIterations(5)
            .run("Assign_100k_256d_1024c", [&]() {
                auto assignments = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>::Assign(
                    data.data(), centroids.data(), n, n_clusters
                );
                ankerl::nanobench::doNotOptimizeAway(assignments);
            });
    }

    std::cout << std::endl;

    // Benchmark with different sampling fractions
    std::cout << "--- Benchmark: Sampling Fraction Impact ---" << std::endl;
    {
        const size_t n = 500000;
        const size_t d = 512;
        const size_t n_clusters = 2000;
        const size_t n_iters = 20;

        std::vector<float> data = make_blobs(n, d, n_clusters);

        std::vector<float> sampling_fractions = {0.1f, 0.25f, 0.5f, 1.0f};

        for (float sampling_fraction : sampling_fractions) {
            skmeans::SuperKMeansConfig config;
            config.iters = n_iters;
            config.sampling_fraction = sampling_fraction;
            config.verbose = false;
            config.n_threads = THREADS;

            std::string bench_name = "Sampling_" + std::to_string(int(sampling_fraction * 100)) + "pct";

            ankerl::nanobench::Bench()
                .epochs(1)
                .epochIterations(1)
                .run(bench_name, [&]() {
                    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                        n_clusters, d, config
                    );
                    auto centroids = kmeans.Train(data.data(), n);
                    ankerl::nanobench::doNotOptimizeAway(centroids);
                });
        }
    }

    std::cout << std::endl << "=== Benchmarks Complete ===" << std::endl;
    return 0;
}
