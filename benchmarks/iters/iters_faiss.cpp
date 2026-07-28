#include <faiss/utils/utils.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "bench_utils.h"

// Iterations benchmark for FAISS: trains on the FULL dataset for a fixed number of
// iterations (1..10), one fresh run per iteration count, and records the full recall /
// vectors_explored curve over ITERS_EXPLORE_FRACTIONS (0.10..20.00). Output:
// iters_faiss.csv.
int main(int argc, char* argv[]) {
    const std::string algorithm = "faiss";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("fmnist");
    std::string experiment_name = (argc > 2) ? std::string(argv[2]) : std::string("iters_faiss");
    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);
    const size_t n_queries = bench_utils::N_QUERIES;

    std::cout << "=== Running algorithm: " << algorithm << " (iterations 1-10) ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << " threads=" << THREADS << std::endl;

    std::vector<float> data;
    std::vector<float> queries;
    data.reserve(n * d);
    queries.reserve(n_queries * d);
    std::ifstream file(bench_utils::get_data_path(dataset), std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open data file\n";
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    file.close();

    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::string queries_filename = bench_utils::get_query_path(dataset);
    std::ifstream gt_check(gt_filename);
    std::ifstream queries_file(queries_filename, std::ios::binary);
    bool have_recall = gt_check.good() && queries_file.good();
    std::unordered_map<int, std::vector<int>> gt_map;
    if (have_recall) {
        gt_check.close();
        queries_file.read(reinterpret_cast<char*>(queries.data()), n_queries * d * sizeof(float));
        queries_file.close();
        gt_map = bench_utils::parse_ground_truth_json(gt_filename);
    } else {
        std::cout << "Ground truth/queries missing; CSV output will be skipped." << std::endl;
    }

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset
    );
    const bool spherical = (is_angular != bench_utils::ANGULAR_DATASETS.end());

    for (int n_iters : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
        std::cout << "\n======== iterations = " << n_iters << " ========" << std::endl;

        faiss::IndexFlatL2 index(d);
        faiss::ClusteringParameters cp;
        cp.niter = n_iters;
        cp.verbose = false;
        cp.max_points_per_centroid = 999999; // no sampling
        cp.nredo = 1;
        cp.spherical = spherical;
        faiss::Clustering clus(d, n_clusters, cp);

        bench_utils::TicToc timer;
        timer.Tic();
        clus.train(n, data.data(), index);
        timer.Toc();
        double construction_time_ms = timer.GetMilliseconds();
        int actual_iterations = static_cast<int>(clus.iteration_stats.size());
        double final_objective = clus.iteration_stats.back().obj;
        std::cout << "Trained in " << construction_time_ms << " ms, objective=" << final_objective
                  << std::endl;

        if (!have_recall) {
            continue;
        }

        std::vector<faiss::idx_t> assignments(n);
        std::vector<float> distances_to_centroids(n);
        const float* centroids = clus.centroids.data();
        faiss::IndexFlatL2 centroid_index(d);
        centroid_index.add(n_clusters, centroids);
        centroid_index.search(n, data.data(), 1, distances_to_centroids.data(), assignments.data());

        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids, n_queries, n_clusters, d, 10,
            bench_utils::ITERS_EXPLORE_FRACTIONS
        );
        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids, n_queries, n_clusters, d, 100,
            bench_utils::ITERS_EXPLORE_FRACTIONS
        );

        std::unordered_map<std::string, std::string> config_map;
        config_map["niter"] = std::to_string(cp.niter);
        config_map["nredo"] = std::to_string(cp.nredo);
        config_map["max_points_per_centroid"] = std::to_string(cp.max_points_per_centroid);
        config_map["seed"] = std::to_string(cp.seed);
        config_map["spherical"] = cp.spherical ? "true" : "false";

        bench_utils::write_results_to_csv(
            experiment_name, algorithm, dataset, n_iters, actual_iterations, static_cast<int>(d), n,
            static_cast<int>(n_clusters), construction_time_ms, static_cast<int>(THREADS),
            final_objective, config_map, results_knn_10, results_knn_100, "",
            bench_utils::ITERS_EXPLORE_FRACTIONS
        );
    }
    return 0;
}
