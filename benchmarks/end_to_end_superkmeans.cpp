#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"
#include "bench_utils.h"

int main(int argc, char* argv[]) {
    // Choose dataset by name. You can also pass the dataset name as the first CLI argument.
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("openai");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, arxiv, sift, fmnist\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t n_queries = 1000;
    const size_t d = it->second.second;
    const size_t n_clusters =
        std::max<size_t>(1u, static_cast<size_t>(std::sqrt(static_cast<double>(n)) * 4.0));
    int n_iters = 2;
    float sampling_fraction = 1.0;
    std::string path_root = std::string(CMAKE_SOURCE_DIR) + "/benchmarks";
    std::string filename = path_root + "/data_" + dataset + ".bin";
    std::string filename_queries = path_root + "/data_" + dataset + "_test.bin";
    constexpr size_t THREADS = 10;
    omp_set_num_threads(THREADS);

    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> data;
    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> queries;
    try {
        data.resize(n * d);
        queries.resize(n_queries * d);
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

    std::ifstream file_queries(filename_queries, std::ios::binary);
    if (!file_queries) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file_queries.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
    file_queries.close();

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.n_threads = THREADS;
    config.objective_k = 10;
    config.ann_explore_fraction = 0.01f;
    config.unrotate_centroids = true;
    config.perform_assignments = false;
    config.early_termination = false;
    config.sampling_fraction = sampling_fraction;
    config.use_blas_only = false;

    auto kmeans_state = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );

    std::vector<float> centroids;
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("SKMeans Queries", [&]() {
        centroids = kmeans_state.Train(data.data(), n, queries.data(), n_queries);
    });
    // ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("SKMeans Queries Sampled", [&]()
    // {
    //     auto centroids = kmeans_state.Train(data.data(), n, nullptr, n_queries, true);
    // });

    // Compute recall if ground truth file exists
    std::string gt_filename = path_root + "/" + dataset + ".json";

    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file_check(filename_queries, std::ios::binary);

    if (gt_file.good() && queries_file_check.good()) {
        gt_file.close();
        queries_file_check.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;
        std::cout << "Queries file: " << filename_queries << std::endl;

        // Load ground truth
        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        std::cout << "Loaded " << gt_map.size() << " queries from ground truth" << std::endl;

        // Assign each data point to its nearest centroid using SuperKMeans::Assign()
        auto assignments = kmeans_state.Assign(
            data.data(), centroids.data(), n, n_clusters
        );

        // Compute recall for different KNN values
        for (int knn : bench_utils::KNN_VALUES) {
            auto results = bench_utils::compute_recall(
                gt_map, assignments, queries.data(), centroids.data(),
                n_queries, n_clusters, d, knn
            );
            bench_utils::print_recall_results(results, knn);
        }
    } else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file_check.good()) {
            std::cout << "Queries file not found: " << filename_queries << std::endl;
        }
    }
}
