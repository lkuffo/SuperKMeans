#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/superkmeans.h"

int main(int argc, char* argv[]) {
    const std::string algorithm = "superkmeans_matryoshka";
    // Usage: <dataset> <target_d> [pruning]
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("yahoo");
    size_t target_d = (argc > 2) ? std::stoull(argv[2]) : 256;
    bool blas_only = !(argc > 3 && std::string(argv[3]) == "pruning");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    int n_iters = 10;
    float sampling_fraction = 1.0f;
    std::string filename = bench_utils::get_data_path(dataset);
    std::string filename_queries = bench_utils::get_query_path(dataset);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    if (target_d >= d) {
        std::cerr << "target_d (" << target_d << ") must be < d (" << d << ")\n";
        return 1;
    }

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d
              << ", target_d=" << target_d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";
    std::cout << "Centroid recovery: recompute as mean of full-d vectors\n";

    // ── Load data ──
    std::vector<float> data(n * d);
    std::vector<float> queries(n_queries * d);

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

    // ── Truncate data to first target_d dimensions (matryoshka prefix) ──
    std::cout << "\nTruncating to first " << target_d << " dimensions (matryoshka)..." << std::endl;
    bench_utils::TicToc truncate_timer;
    truncate_timer.Tic();
    std::vector<float> truncated_data(n * target_d);
    for (size_t i = 0; i < n; ++i) {
        std::copy_n(data.data() + i * d, target_d, truncated_data.data() + i * target_d);
    }
    truncate_timer.Toc();
    std::cout << "Truncation completed in " << truncate_timer.GetMilliseconds() << " ms\n";

    // ── Run SuperKMeans on truncated data ──
    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = sampling_fraction;
    config.use_blas_only = blas_only;
    config.data_already_rotated = false;
    config.tol = 1e-3f;

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means" << std::endl;
        config.angular = true;
    }

    auto kmeans =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, target_d, config
        );
    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> truncated_centroids = kmeans.Train(truncated_data.data(), n);
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();
    int actual_iterations = static_cast<int>(kmeans.iteration_stats.size());
    double final_objective = kmeans.iteration_stats.back().objective;

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;
    std::cout << "Final objective (in truncated space): " << final_objective << std::endl;

    // ── Recover full-dimensional centroids ──
    // Assign in truncated space, then compute mean of full-d vectors per cluster
    std::cout << "\nComputing assignments in truncated space..." << std::endl;
    auto assignments = kmeans.Assign(
        truncated_data.data(), truncated_centroids.data(), n, n_clusters
    );

    std::cout << "Recomputing centroids as mean of full-d vectors..." << std::endl;
    std::vector<float> centroids(n_clusters * d, 0.0f);
    std::vector<size_t> cluster_sizes(n_clusters, 0);

    for (size_t i = 0; i < n; ++i) {
        const uint32_t c = assignments[i];
        cluster_sizes[c]++;
        for (size_t j = 0; j < d; ++j) {
            centroids[c * d + j] += data[i * d + j];
        }
    }
    for (size_t c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] > 0) {
            const float inv_size = 1.0f / static_cast<float>(cluster_sizes[c]);
            for (size_t j = 0; j < d; ++j) {
                centroids[c * d + j] *= inv_size;
            }
        }
    }

    // ── Compute full-d assignments and WCSS ──
    skmeans::SuperKMeansConfig assign_config;
    assign_config.n_threads = THREADS;
    assign_config.use_blas_only = true;
    auto kmeans_assign =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, assign_config
        );
    auto assignments_full = kmeans_assign.Assign(data.data(), centroids.data(), n, n_clusters);

    // Compute WCSS in full-d space
    double wcss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t c = assignments_full[i];
        for (size_t j = 0; j < d; ++j) {
            double diff = data[i * d + j] - centroids[c * d + j];
            wcss += diff * diff;
        }
    }
    std::cout << "\nFull-d WCSS: " << wcss << std::endl;

    auto balance_stats =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>::
            GetClustersBalanceStats(assignments_full.data(), n, n_clusters);
    balance_stats.print();

    // ── Compute recall ──
    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file_check(filename_queries, std::ios::binary);
    if (gt_file.good() && queries_file_check.good()) {
        gt_file.close();
        queries_file_check.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;
        std::cout << "Queries file: " << filename_queries << std::endl;

        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        // Recall is computed using full-d centroids and full-d queries
        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments_full, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_knn_10, 10);

        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments_full, queries.data(), centroids.data(), n_queries, n_clusters, d,
            100
        );
        bench_utils::print_recall_results(results_knn_100, 100);
    } else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file_check.good()) {
            std::cout << "Queries file not found: " << filename_queries << std::endl;
        }
        std::cout << "Skipping recall computation (requires ground truth)" << std::endl;
    }
}
