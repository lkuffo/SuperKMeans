#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <faiss/utils/utils.h>

#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "bench_utils.h"
#include "superkmeans/nanobench.h"

int main(int argc, char* argv[]) {
    // Experiment configuration
    const std::string algorithm = "faiss";
    const std::string experiment_name = "varying_k";

    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("openai");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, wiki, arxiv, sift, fmnist\n";
        return 1;
    }

    const int n = it->second.first;
    const int d = it->second.second;
    int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);
    std::string filename = bench_utils::get_data_path(dataset);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "Experiment: " << experiment_name << std::endl;
    std::cout << "Compile options: " << faiss::get_compile_options() << std::endl;

    std::vector<float> data;
    try {
        data.resize(n * d);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what()
                  << "\n";
        return 1;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    // Loop over different n_clusters values
    for (int n_clusters : bench_utils::VARYING_K_VALUES) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "n_clusters=" << n_clusters << std::endl;
        std::cout << "========================================" << std::endl;

        faiss::IndexFlatL2 index(d);

        // Set up clustering parameters
        faiss::ClusteringParameters cp;
        cp.niter = n_iters;
        cp.verbose = true;
        cp.max_points_per_centroid = 999999; // We don't want to take samples
        cp.nredo = 1;

        // Create the clustering object
        faiss::Clustering clus(d, n_clusters, cp);

        // Time the training
        bench_utils::TicToc timer;
        timer.Tic();
        clus.train(n, data.data(), index);
        timer.Toc();
        double construction_time_ms = timer.GetMilliseconds();

        // Get actual iterations and final objective
        int actual_iterations = static_cast<int>(clus.iteration_stats.size());
        double final_objective = clus.iteration_stats.back().obj;

        std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
        std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
                  << std::endl;
        std::cout << "Final objective: " << final_objective << std::endl;

        // Skip assignment and recall computation for this benchmark
        // We only care about training time and objective for the varying_k parameter sweep
        std::vector<std::tuple<int, float, float, float, float>> results_knn_10;   // Empty
        std::vector<std::tuple<int, float, float, float, float>> results_knn_100;  // Empty

        // Create config dictionary with FAISS parameters
        std::unordered_map<std::string, std::string> config_map;
        config_map["niter"] = std::to_string(cp.niter);
        config_map["nredo"] = std::to_string(cp.nredo);
        config_map["max_points_per_centroid"] = std::to_string(cp.max_points_per_centroid);
        config_map["min_points_per_centroid"] = std::to_string(cp.min_points_per_centroid);
        config_map["seed"] = std::to_string(cp.seed);
        config_map["spherical"] = cp.spherical ? "true" : "false";
        config_map["int_centroids"] = cp.int_centroids ? "true" : "false";
        config_map["update_index"] = cp.update_index ? "true" : "false";
        config_map["frozen_centroids"] = cp.frozen_centroids ? "true" : "false";
        config_map["verbose"] = cp.verbose ? "true" : "false";

        // Write results to CSV
        bench_utils::write_results_to_csv(
            experiment_name,
            algorithm,
            dataset,
            n_iters,
            actual_iterations,
            d,
            n,
            n_clusters,
            construction_time_ms,
            static_cast<int>(THREADS),
            final_objective,
            config_map,
            results_knn_10,
            results_knn_100
        );
    }
}
