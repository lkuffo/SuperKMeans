#ifndef SMALLER_TEST
#define SMALLER_TEST = true
#endif

#include <faiss/utils/utils.h>

#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include "bench_utils.h"

int main() {
    const std::string experiment_name = "blog_post_1";
    const std::string algorithm = "faiss_ivfflat";

#ifdef SMALLER_TEST
    const std::string dataset = "openai";
    const size_t n = 999000;
    const size_t n_queries = 1000;
    const size_t d = 1536;
    const size_t n_clusters = 10000;
    const std::string filename = "./data/data_openai.bin";
    const std::string filename_queries = "./data/data_openai_test.bin";
#else
    const std::string dataset = "cohere50m";
    const size_t n = 50000000;
    const size_t n_queries = 1000;
    const size_t d = 1024;
    const size_t n_clusters = 200000;
    const std::string filename = "./data/data_cohere50m.bin";
    const std::string filename_queries = "./data/data_cohere50m_test.bin";
#endif

    const float sampling_fraction = 0.3f;
    const int max_points_per_centroid = static_cast<int>((n * sampling_fraction) / n_clusters);

    const size_t threads = omp_get_max_threads();
    omp_set_num_threads(threads);

    std::cout << "=== Running FAISS IVFFlat on Cohere 50M dataset ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters
              << " max_points_per_centroid=" << max_points_per_centroid
              << " (equivalent to sampling_fraction=" << sampling_fraction << ")\n";
    std::cout << "Compile options: " << faiss::get_compile_options() << std::endl;

    std::vector<float> data;
    data.reserve(n * d);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    file.close();

    std::vector<float> queries;
    queries.reserve(n_queries * d);
    std::ifstream file_queries(filename_queries, std::ios::binary);
    if (!file_queries) {
        std::cerr << "Failed to open " << filename_queries << std::endl;
        return 1;
    }
    file_queries.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    file_queries.close();

    // Build IVFFlat index
    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFFlat index(&quantizer, d, n_clusters);
    index.cp.max_points_per_centroid = max_points_per_centroid;
    index.cp.verbose = true;

    // Train (k-means on sampled data)
    bench_utils::TicToc timer;
    timer.Tic();
    index.train(n, data.data());
    timer.Toc();
    double train_time_ms = timer.GetMilliseconds();
    std::cout << "\nTraining completed in " << train_time_ms << " ms" << std::endl;

    // Add all vectors (assignment)
    timer.Reset();
    timer.Tic();
    index.add(n, data.data());
    timer.Toc();
    double add_time_ms = timer.GetMilliseconds();
    std::cout << "Add (assignment) completed in " << add_time_ms << " ms" << std::endl;
    std::cout << "Total (train + add): " << (train_time_ms + add_time_ms) << " ms" << std::endl;

    // Compute recall if ground truth exists
    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::ifstream gt_file(gt_filename);
    if (gt_file.good()) {
        gt_file.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;

        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        // Get assignments by searching for 1-NN against centroids
        std::vector<faiss::idx_t> assignments(n);
        std::vector<float> distances_to_centroids(n);
        const float* centroids = quantizer.get_xb();
        quantizer.search(n, data.data(), 1, distances_to_centroids.data(), assignments.data());

        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids, n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_knn_10, 10);

        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids, n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(results_knn_100, 100);

        std::unordered_map<std::string, std::string> config_map;
        config_map["niter"] = std::to_string(index.cp.niter);
        config_map["nredo"] = std::to_string(index.cp.nredo);
        config_map["max_points_per_centroid"] = std::to_string(index.cp.max_points_per_centroid);
        config_map["sampling_fraction"] = std::to_string(sampling_fraction);
        config_map["verbose"] = index.cp.verbose ? "true" : "false";
        config_map["training_time"] = std::to_string(train_time_ms);
        config_map["assignment_time"] = std::to_string(add_time_ms);

        bench_utils::write_results_to_csv(
            experiment_name,
            algorithm,
            dataset,
            index.cp.niter,
            index.cp.niter,
            static_cast<int>(d),
            n,
            static_cast<int>(n_clusters),
            train_time_ms + add_time_ms,
            static_cast<int>(threads),
            0.0,
            config_map,
            results_knn_10,
            results_knn_100
        );
    } else {
        std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        std::cout << "Skipping recall computation" << std::endl;
    }
}
