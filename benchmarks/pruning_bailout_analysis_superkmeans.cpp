#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

int main(int argc, char* argv[]) {
    // Experiment configuration
    const std::string experiment_name = "pruning_bailout_analysis";
    const std::string algorithm = "superkmeans";
    const int n_iters = 10;
    const float sampling_fraction = 1.0f;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    std::cout << "=== Running pruning_bailout_analysis benchmark ===" << std::endl;
    std::cout << "Fixed parameters: iters=" << n_iters << ", sampling_fraction=" << sampling_fraction
              << std::endl;
    std::cout << "Threads: " << THREADS << std::endl;
    std::cout << "Eigen # threads: " << Eigen::nbThreads()
              << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    // Two modes to test
    std::vector<std::pair<std::string, bool>> test_modes = {
        {"with_pruning", false},  // use_blas_only = false (pruning enabled)
        {"blas_only", true}       // use_blas_only = true (no pruning)
    };

    std::cout << "Testing " << test_modes.size() << " modes: with_pruning vs blas_only" << std::endl;

    // Loop over all datasets
    for (const auto& dataset_entry : bench_utils::DATASET_PARAMS) {
        const std::string& dataset = dataset_entry.first;
        const size_t n = dataset_entry.second.first;
        const size_t d = dataset_entry.second.second;
        const size_t n_clusters =
            std::max<size_t>(1u, static_cast<size_t>(std::sqrt(static_cast<double>(n)) * 4.0));
        std::string filename = bench_utils::get_data_path(dataset);

        std::cout << "\n========================================" << std::endl;
        std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")" << std::endl;
        std::cout << "n_clusters=" << n_clusters << std::endl;
        std::cout << "========================================" << std::endl;

        // Load data
        std::vector<skmeans::skmeans_value_t<skmeans::Quantization::f32>> data;
        try {
            data.resize(n * d);
        } catch (const std::bad_alloc& e) {
            std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what()
                      << "\n";
            std::cerr << "Skipping dataset: " << dataset << std::endl;
            continue;
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open data file: " << filename << std::endl;
            std::cerr << "Skipping dataset: " << dataset << std::endl;
            continue;
        }
        file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
        file.close();

        // Warmup iteration - run once per dataset to warm up caches
        std::cout << "\n--- Warmup iteration (not measured) ---" << std::endl;
        {
            skmeans::SuperKMeansConfig warmup_config;
            warmup_config.iters = 1;  // Single iteration for warmup
            warmup_config.verbose = false;
            warmup_config.n_threads = THREADS;
            warmup_config.objective_k = 100;
            warmup_config.ann_explore_fraction = 0.01f;
            warmup_config.unrotate_centroids = true;
            warmup_config.perform_assignments = false;
            warmup_config.early_termination = false;
            warmup_config.sampling_fraction = sampling_fraction;
            warmup_config.use_blas_only = false;

            auto is_angular = std::find(
                bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
            );
            if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
                warmup_config.angular = true;
            }

            auto warmup_state =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, warmup_config
                );
            std::vector<float> warmup_centroids = warmup_state.Train(data.data(), n);
            std::cout << "Warmup completed" << std::endl;
        }

        // Loop over test modes (with_pruning vs blas_only)
        for (const auto& [mode_name, use_blas_only] : test_modes) {
            std::cout << "\n--- Testing mode: " << mode_name
                      << " (use_blas_only=" << (use_blas_only ? "true" : "false") << ") ---"
                      << std::endl;

            skmeans::SuperKMeansConfig config;
            config.iters = n_iters;
            config.verbose = true;
            config.n_threads = THREADS;
            config.objective_k = 100;
            config.ann_explore_fraction = 0.01f;
            config.unrotate_centroids = true;
            config.perform_assignments = false;
            config.early_termination = false;
            config.sampling_fraction = sampling_fraction;
            config.use_blas_only = use_blas_only;

            auto is_angular = std::find(
                bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
            );
            if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
                std::cout << "Using spherical k-means" << std::endl;
                config.angular = true;
            }

            auto kmeans_state =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, config
                );

            // Time the training
            bench_utils::TicToc timer;
            timer.Tic();
            std::vector<float> centroids = kmeans_state.Train(data.data(), n);
            timer.Toc();
            double construction_time_ms = timer.GetMilliseconds();

            // Get actual iterations and final objective
            int actual_iterations = static_cast<int>(kmeans_state.iteration_stats.size());
            double final_objective = kmeans_state.iteration_stats.back().objective;

            std::cout << "Training: " << construction_time_ms << " ms, "
                      << "Objective: " << final_objective << ", "
                      << "Iterations: " << actual_iterations << std::endl;

            // Skip assignment and recall computation for this benchmark
            std::vector<std::tuple<int, float, float, float, float>> results_knn_10;   // Empty
            std::vector<std::tuple<int, float, float, float, float>> results_knn_100;  // Empty

            // Create comprehensive config dictionary with all parameters
            std::unordered_map<std::string, std::string> config_map;
            config_map["iters"] = std::to_string(config.iters);
            config_map["sampling_fraction"] = std::to_string(config.sampling_fraction);
            config_map["n_threads"] = std::to_string(config.n_threads);
            config_map["seed"] = std::to_string(config.seed);
            config_map["use_blas_only"] = config.use_blas_only ? "true" : "false";
            config_map["tol"] = std::to_string(config.tol);
            config_map["recall_tol"] = std::to_string(config.recall_tol);
            config_map["early_termination"] = config.early_termination ? "true" : "false";
            config_map["sample_queries"] = config.sample_queries ? "true" : "false";
            config_map["objective_k"] = std::to_string(config.objective_k);
            config_map["ann_explore_fraction"] = std::to_string(config.ann_explore_fraction);
            config_map["unrotate_centroids"] = config.unrotate_centroids ? "true" : "false";
            config_map["perform_assignments"] = config.perform_assignments ? "true" : "false";
            config_map["verbose"] = config.verbose ? "true" : "false";
            config_map["mode"] = "\"" + mode_name + "\"";

            // Write results to CSV
            bench_utils::write_results_to_csv(
                experiment_name,
                algorithm,
                dataset,
                n_iters,
                actual_iterations,
                static_cast<int>(d),
                n,
                static_cast<int>(n_clusters),
                construction_time_ms,
                static_cast<int>(THREADS),
                final_objective,
                config_map,
                results_knn_10,
                results_knn_100
            );
        }  // End modes loop
    }      // End dataset loop

    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << "Results written to: results/pruning_bailout_analysis.csv" << std::endl;
    std::cout << "========================================" << std::endl;
}
