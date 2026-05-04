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
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("mxbai");
    std::string pq_variant = (argc > 2) ? std::string(argv[2]) : std::string("pq8");
    uint32_t pq_m = (argc > 3) ? static_cast<uint32_t>(std::stoi(argv[3])) : 16;

    if (pq_variant != "pq8" && pq_variant != "pq4") {
        std::cerr << "Invalid PQ variant '" << pq_variant << "' (expected: pq8, pq4)\n";
        return 1;
    }

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t n_queries = bench_utils::N_QUERIES;
    const size_t d = it->second.second;

    if (d % pq_m != 0) {
        std::cerr << "d=" << d << " is not divisible by M=" << pq_m << "\n";
        return 1;
    }

    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    int n_iters = 10;
    std::string filename = bench_utils::get_data_path(dataset);
    std::string filename_queries = bench_utils::get_query_path(dataset);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    std::cout << "=== PQ K-Means: " << pq_variant << " (M=" << pq_m
              << ", dsub=" << d / pq_m << ") ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " n_iters=" << n_iters << "\n";

    // ── Load data and queries ──
    std::vector<float> data(n * d);
    std::vector<float> queries(n_queries * d);

    {
        std::ifstream f(filename, std::ios::binary);
        if (!f) { std::cerr << "Failed to open " << filename << std::endl; return 1; }
        f.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }
    {
        std::ifstream f(filename_queries, std::ios::binary);
        if (!f) { std::cerr << "Failed to open " << filename_queries << std::endl; return 1; }
        f.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
    }

    // ── Configure and Train ──
    using SKM = skmeans::SuperKMeans<skmeans::Quantization::u8, skmeans::DistanceFunction::l2>;

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = true;
    config.verbose_detail = true;
    config.n_threads = THREADS;
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.tol = 1e-3f;
    config.use_blas_only = true;
    config.quantized_centroid_update = true;
    config.data_already_rotated = true;
    config.pq_m = pq_m;
    config.quantizer_type = (pq_variant == "pq8")
        ? skmeans::QuantizerType::pq8
        : skmeans::QuantizerType::pq4;
    config.full_precision_final_centroids = false;

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means" << std::endl;
        config.angular = true;
    }

    auto kmeans = SKM(n_clusters, d, config);

    bench_utils::TicToc timer;
    timer.Tic();
    std::vector<float> centroids = kmeans.Train(data.data(), n);
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();
    int actual_iterations = static_cast<int>(kmeans.iteration_stats.size());
    double final_objective = kmeans.iteration_stats.back().objective;

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;
    std::cout << "Final objective (quantized): " << final_objective << std::endl;

    // ── Assignments ──
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    auto q_assignments = kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);

    // Internal assignments from last iteration (no re-encoding round-trip)
    std::vector<uint32_t> internal_assignments(
        kmeans.assignments.get(), kmeans.assignments.get() + n
    );

    double wcss_assign = SKM::ComputeWCSS(
        data.data(), centroids.data(), assignments.data(), n, d
    );
    double wcss_q_assign = SKM::ComputeWCSS(
        data.data(), centroids.data(), q_assignments.data(), n, d
    );
    double wcss_internal = SKM::ComputeWCSS(
        data.data(), centroids.data(), internal_assignments.data(), n, d
    );
    std::cout << "WCSS (f32, Assign):          " << std::fixed << std::setprecision(2)
              << wcss_assign << std::endl;
    std::cout << "WCSS (f32, QuantizedAssign): " << std::fixed << std::setprecision(2)
              << wcss_q_assign << std::endl;
    std::cout << "WCSS (f32, InternalAssign):  " << std::fixed << std::setprecision(2)
              << wcss_internal << std::endl;

    // Compare QuantizedAssign vs InternalAssign (should be identical if no round-trip error)
    size_t n_differ = 0;
    for (size_t i = 0; i < n; ++i) {
        if (q_assignments[i] != internal_assignments[i]) ++n_differ;
    }
    std::cout << "QuantizedAssign vs InternalAssign differ: " << n_differ
              << " / " << n << " (" << std::setprecision(4)
              << (100.0 * n_differ / n) << "%)" << std::endl;

    std::cout << "\n--- Assign() cluster balance ---" << std::endl;
    SKM::GetClustersBalanceStats(assignments.data(), n, n_clusters).print();

    std::cout << "--- QuantizedAssign() cluster balance ---" << std::endl;
    SKM::GetClustersBalanceStats(q_assignments.data(), n, n_clusters).print();

    // ── Recall ──
    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file_check(filename_queries, std::ios::binary);
    if (gt_file.good() && queries_file_check.good()) {
        gt_file.close();
        queries_file_check.close();

        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        std::cout << "\n  [Assign()]" << std::endl;
        auto results_10 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_10, 10);

        auto results_100 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(results_100, 100);

        std::cout << "\n  [QuantizedAssign()]" << std::endl;
        auto q_results_10 = bench_utils::compute_recall(
            gt_map, q_assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(q_results_10, 10);

        auto q_results_100 = bench_utils::compute_recall(
            gt_map, q_assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(q_results_100, 100);

        std::cout << "\n  [InternalAssign() — last iteration, no re-encoding]" << std::endl;
        auto i_results_10 = bench_utils::compute_recall(
            gt_map, internal_assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(i_results_10, 10);

        auto i_results_100 = bench_utils::compute_recall(
            gt_map, internal_assignments, queries.data(), centroids.data(), n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(i_results_100, 100);
    } else {
        if (!gt_file.good())
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        if (!queries_file_check.good())
            std::cout << "Queries file not found: " << filename_queries << std::endl;
    }
}
