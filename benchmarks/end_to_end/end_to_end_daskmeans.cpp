// End-to-end benchmark for Dask-means (arXiv 2412.02244).
// Uses the vendored implementation in extern/daskmeans as-is (double->float, seed=42,
// MAX_ITERATIONS=25, OpenMP added). Dask-means never fills a label array, so per-point
// assignments are recovered with a final (parallel) 1-NN pass against the final centroids.
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_utils.h"

#include "Centroid.h"
#include "DaskMeans.h"

int main(int argc, char* argv[]) {
    const std::string algorithm = "daskmeans";
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

    // leaf_capacity=30 (as in their harness), max_iterations=25, convergence_threshold=0 (fixed iters)
    DaskMeans km(30, n_iters, 0.0f);
    km.initParameters(static_cast<int>(n), static_cast<int>(d), static_cast<int>(n_clusters));
    km.dataset.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const float* row = data.data() + i * d;
        km.dataset[i].assign(row, row + d);
    }

    bench_utils::TicToc timer;
    timer.Tic();
    km.run();
    timer.Toc();
    const double construction_time_ms = timer.GetMilliseconds();
    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;

    // Extract centroids (k x d, row-major float).
    std::vector<float> centroids(n_clusters * d);
    for (size_t c = 0; c < n_clusters; ++c) {
        const std::vector<float>& coord = km.centroid_list[c]->coordinate;
        std::copy(coord.begin(), coord.end(), centroids.begin() + c * d);
    }

    // Dask-means does not expose per-point labels; recover them with a parallel 1-NN pass.
    std::vector<int> assignments(n);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const float* x = data.data() + i * d;
        float best = std::numeric_limits<float>::max();
        int best_c = 0;
        for (size_t c = 0; c < n_clusters; ++c) {
            const float* cc = centroids.data() + c * d;
            float dist = 0.0f;
            for (size_t j = 0; j < d; ++j) {
                const float diff = x[j] - cc[j];
                dist += diff * diff;
            }
            if (dist < best) {
                best = dist;
                best_c = static_cast<int>(c);
            }
        }
        assignments[i] = best_c;
    }

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
        config_map["leaf_capacity"] = "30";
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
