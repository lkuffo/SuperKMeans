// End-to-end benchmark for Marigold k-means (Mortensen et al., VLDB 2023).
// Uses the vendored implementation in extern/marigold as-is (double->float, seed=42,
// OpenMP on the assignment loop); no vector padding — the algorithm clusters on its
// first floor(sqrt(d))^2 dims by design.
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_utils.h"
#include "strategies/marigold_kmeans_strategy.cpp"

int main(int argc, char* argv[]) {
    const std::string algorithm = "marigold";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("fmnist");
    std::string experiment_name = (argc > 2) ? std::string(argv[2]) : std::string("end_to_end");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    const int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(static_cast<int>(THREADS));
    const std::string filename = bench_utils::get_data_path(dataset);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters << " threads=" << THREADS
              << std::endl;

    std::vector<float> data(n * d);
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open " << filename << std::endl;
            return 1;
        }
        file.read(
            reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * d) * sizeof(float)
        );
    }

    Dataset ds(static_cast<int>(n), static_cast<int>(d), data.data()); // borrows data
    MARIGOLDKmeansStrategy mg;
    bench_utils::TicToc timer;
    timer.Tic();
    mg.init(
        n_iters, static_cast<int>(n), static_cast<int>(d), static_cast<int>(n_clusters), &ds
    );
    int* labels_ptr = mg.run(&ds);
    timer.Toc();
    const double construction_time_ms = timer.GetMilliseconds();

    // Copy results out BEFORE clear() frees the strategy's buffers.
    std::vector<int> assignments(labels_ptr, labels_ptr + n);
    const float* mg_centroids = mg.get_centroids();
    std::vector<float> centroids(mg_centroids, mg_centroids + n_clusters * d);
    mg.clear();

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;

    const std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    const std::string queries_filename = bench_utils::get_query_path(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file(queries_filename, std::ios::binary);
    if (gt_file.good() && queries_file.good()) {
        gt_file.close();
        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        const int n_queries = bench_utils::N_QUERIES;
        std::vector<float> queries(n_queries * d);
        queries_file.read(
            reinterpret_cast<char*>(queries.data()),
            static_cast<std::streamsize>(n_queries * d) * sizeof(float)
        );
        queries_file.close();

        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_knn_10, 10);
        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(results_knn_100, 100);

        std::cout << "\n--- Computing Internal Metrics ---" << std::endl;
        auto internal_metrics = bench_utils::compute_internal_metrics(
            data.data(), centroids.data(), assignments, n, n_clusters, d
        );
        std::cout << "Final objective (WCSS): " << internal_metrics.wgss << std::endl;
        std::cout << "Calinski-Harabasz: " << internal_metrics.calinski_harabasz
                  << "  Silhouette: " << internal_metrics.silhouette << std::endl;

        std::unordered_map<std::string, std::string> config_map;
        config_map["iters"] = std::to_string(n_iters);
        config_map["seed"] = "42";
        config_map["precision"] = "\"float32\"";
        config_map["calinski_harabasz"] = std::to_string(internal_metrics.calinski_harabasz);
        config_map["silhouette"] = std::to_string(internal_metrics.silhouette);

        bench_utils::write_results_to_csv(
            experiment_name,
            algorithm,
            dataset,
            n_iters,
            n_iters,
            static_cast<int>(d),
            n,
            static_cast<int>(n_clusters),
            construction_time_ms,
            static_cast<int>(THREADS),
            internal_metrics.wgss,
            config_map,
            results_knn_10,
            results_knn_100
        );
    } else {
        std::cout << "Ground truth or queries file not found; skipping CSV output" << std::endl;
    }
    return 0;
}
