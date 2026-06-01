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

template <skmeans::Quantization Q>
static void RunHNSW(
    const std::string& dataset,
    const float* data, size_t n, size_t d,
    const float* queries, size_t n_queries,
    size_t n_clusters,
    const skmeans::SuperKMeansConfig& config
) {
    using SKM = skmeans::SuperKMeans<Q, skmeans::DistanceFunction::l2>;

    auto kmeans_state = SKM(n_clusters, d, config);
    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> centroids = kmeans_state.Train(data, n);
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();
    int actual_iterations = static_cast<int>(kmeans_state.iteration_stats.size());
    double final_objective = kmeans_state.iteration_stats.back().objective;

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << "\n";
    std::cout << "Final objective: " << final_objective << std::endl;

    // Assign(): brute-force f32 baseline (uses f32_batch_computer, not HNSW)
    auto assignments = kmeans_state.Assign(data, centroids.data(), n, n_clusters);
    // QuantizedAssign(): re-invokes quantizer->FindNearestNeighbor, i.e. the HNSW path
    auto q_assignments = kmeans_state.QuantizedAssign(data, centroids.data(), n, n_clusters);

    using SKM_f32 =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>;
    double wcss_assign = SKM_f32::ComputeWCSS(
        data, centroids.data(), assignments.data(), n, d
    );
    double wcss_q_assign = SKM_f32::ComputeWCSS(
        data, centroids.data(), q_assignments.data(), n, d
    );
    std::cout << "WCSS (f32, Assign):          " << std::fixed << std::setprecision(2)
              << wcss_assign << std::endl;
    std::cout << "WCSS (f32, QuantizedAssign): " << std::fixed << std::setprecision(2)
              << wcss_q_assign << std::endl;

    std::cout << "\n--- Assign() cluster balance ---" << std::endl;
    auto balance_stats = SKM_f32::GetClustersBalanceStats(assignments.data(), n, n_clusters);
    balance_stats.print();

    std::cout << "--- QuantizedAssign() cluster balance ---" << std::endl;
    auto q_balance_stats = SKM_f32::GetClustersBalanceStats(q_assignments.data(), n, n_clusters);
    q_balance_stats.print();

    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::ifstream gt_file(gt_filename);
    if (gt_file.good() && queries != nullptr) {
        gt_file.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);

        std::cout << "\n  [Assign()]" << std::endl;
        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments, queries, centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_knn_10, 10);
        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments, queries, centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(results_knn_100, 100);

        std::cout << "\n  [QuantizedAssign()]" << std::endl;
        auto q_results_knn_10 = bench_utils::compute_recall(
            gt_map, q_assignments, queries, centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(q_results_knn_10, 10);
        auto q_results_knn_100 = bench_utils::compute_recall(
            gt_map, q_assignments, queries, centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(q_results_knn_100, 100);
    } else {
        std::cout << "\nGround truth not found, skipping recall." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    const std::string algorithm = "superkmeans_hnsw";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("yahoo");

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

    int hnsw_M = (argc > 2) ? std::stoi(argv[2]) : 32;
    int hnsw_ef_construction = (argc > 3) ? std::stoi(argv[3]) : 40;
    int hnsw_ef_search = (argc > 4) ? std::stoi(argv[4]) : 16;
    bool hnsw_use_warm_start = (argc > 5) && std::string(argv[5]) == "true";
    // Backend selector: "f32" (default, FAISS f32 HNSW) or "sq8" (USearch SQ8 HNSW)
    std::string hnsw_backend = (argc > 6) ? std::string(argv[6]) : std::string("f32");

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters
              << " sampling_fraction=" << sampling_fraction << "\n";
    std::cout << "HNSW: backend=" << hnsw_backend
              << " M=" << hnsw_M
              << " efConstruction=" << hnsw_ef_construction
              << " efSearch=" << hnsw_ef_search
              << " warm_start=" << (hnsw_use_warm_start ? "true" : "false") << "\n";

    std::vector<float> data;
    std::vector<float> queries;
    try {
        data.reserve(n * d);
        queries.reserve(n_queries * d);
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
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    file.close();

    bool have_queries = false;
    std::ifstream file_queries(filename_queries, std::ios::binary);
    if (file_queries) {
        file_queries.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
        file_queries.close();
        have_queries = true;
    }

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.verbose_detail = true;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = sampling_fraction;
    config.tol = 1e-3f;
    if (hnsw_backend == "sq8") {
        config.quantizer_type = skmeans::QuantizerType::hnsw_sq8;
        // Route centroid updates through the u8 accumulator in HNSWSQ8Quantizer,
        // matching the sq8 vanilla path. Without this the framework falls back
        // to F32Quantizer::UpdateCentroids on the raw float data.
        config.quantized_centroid_update = true;
    } else if (hnsw_backend == "faiss_sq8") {
        config.quantizer_type = skmeans::QuantizerType::hnsw_faiss_sq8;
        config.quantized_centroid_update = true;
    } else {
        config.quantizer_type = skmeans::QuantizerType::hnsw;
    }
    config.hnsw_M = hnsw_M;
    config.hnsw_ef_construction = hnsw_ef_construction;
    config.hnsw_ef_search = hnsw_ef_search;
    config.hnsw_use_warm_start = hnsw_use_warm_start;
    // HNSW does not support pruning; ensure GEMM-only path
    config.use_blas_only = true;
    config.data_already_rotated = true;

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means" << std::endl;
        config.angular = true;
    }

    const float* q_ptr = have_queries ? queries.data() : nullptr;
    if (hnsw_backend == "sq8") {
        RunHNSW<skmeans::Quantization::u8>(
            dataset, data.data(), n, d, q_ptr, n_queries, n_clusters, config
        );
    } else {
        RunHNSW<skmeans::Quantization::f32>(
            dataset, data.data(), n, d, q_ptr, n_queries, n_clusters, config
        );
    }
    return 0;
}
