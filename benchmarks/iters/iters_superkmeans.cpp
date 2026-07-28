#include <algorithm>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

// Iterations benchmark for SuperKMeans: trains on the FULL dataset for a fixed number
// of iterations (1..10), one fresh run per iteration count (SuperKMeans cannot record
// recall per iteration and resume), and records the full recall / vectors_explored
// curve over ITERS_EXPLORE_FRACTIONS (0.10..20.00). Output: iters_superkmeans.csv.
int main(int argc, char* argv[]) {
    const std::string algorithm = "superkmeans";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("fmnist");
    const std::string experiment_name =
        (argc > 2) ? std::string(argv[2]) : std::string("iters_superkmeans");
    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);
    std::string filename = bench_utils::get_data_path(dataset);
    std::string filename_queries = bench_utils::get_query_path(dataset);
    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);

    std::cout << "=== Running algorithm: " << algorithm << " (iterations 1-10) ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " threads=" << THREADS << std::endl;

    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> data;
    std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> queries;
    data.reserve(n * d);
    queries.reserve(n_queries * d);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    file.close();
    std::ifstream file_queries(filename_queries, std::ios::binary);
    if (!file_queries) {
        std::cerr << "Failed to open " << filename_queries << std::endl;
        return 1;
    }
    file_queries.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    file_queries.close();

    for (int n_iters : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
        std::cout << "\n======== iterations = " << n_iters << " ========" << std::endl;

        skmeans::SuperKMeansConfig config;
        config.iters = n_iters;
        config.verbose = false;
        config.n_threads = THREADS;
        config.objective_k = 100;
        config.ann_explore_fraction = 0.01f;
        config.unrotate_centroids = true;
        config.early_termination = false;
        config.sampling_fraction = 1.0f; // full dataset
        config.use_blas_only = false;
        auto is_angular = std::find(
            bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
        );
        if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
            config.angular = true;
        }

        auto kmeans_state =
            skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                n_clusters, d, config
            );

        bench_utils::TicToc timer;
        timer.Tic();
        std::vector<float> centroids = kmeans_state.Train(data.data(), n);
        timer.Toc();
        double construction_time_ms = timer.GetMilliseconds();
        int actual_iterations = static_cast<int>(kmeans_state.iteration_stats.size());
        double final_objective = kmeans_state.iteration_stats.back().objective;
        std::cout << "Trained in " << construction_time_ms << " ms, objective=" << final_objective
                  << " (actual iters=" << actual_iterations << ")" << std::endl;

        std::ifstream gt_file(gt_filename);
        std::ifstream queries_file_check(filename_queries, std::ios::binary);
        if (!gt_file.good() || !queries_file_check.good()) {
            std::cout << "Skipping CSV output (recall requires ground truth + queries)" << std::endl;
            continue;
        }
        gt_file.close();
        queries_file_check.close();

        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        auto assignments = kmeans_state.Assign(data.data(), centroids.data(), n, n_clusters);

        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10,
            bench_utils::ITERS_EXPLORE_FRACTIONS
        );
        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100,
            bench_utils::ITERS_EXPLORE_FRACTIONS
        );

        std::unordered_map<std::string, std::string> config_map;
        config_map["iters"] = std::to_string(config.iters);
        config_map["sampling_fraction"] = std::to_string(config.sampling_fraction);
        config_map["n_threads"] = std::to_string(config.n_threads);
        config_map["seed"] = std::to_string(config.seed);
        config_map["objective_k"] = std::to_string(config.objective_k);
        config_map["ann_explore_fraction"] = std::to_string(config.ann_explore_fraction);
        config_map["unrotate_centroids"] = config.unrotate_centroids ? "true" : "false";
        config_map["angular"] = config.angular ? "true" : "false";

        bench_utils::write_results_to_csv(
            experiment_name, algorithm, dataset, n_iters, actual_iterations, static_cast<int>(d), n,
            static_cast<int>(n_clusters), construction_time_ms, static_cast<int>(THREADS),
            final_objective, config_map, results_knn_10, results_knn_100, "",
            bench_utils::ITERS_EXPLORE_FRACTIONS
        );
    }
    return 0;
}
