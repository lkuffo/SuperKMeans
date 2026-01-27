#include <faiss/utils/utils.h>

#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "bench_utils.h"

int main(int argc, char* argv[]) {
    const std::string algorithm = "faiss";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("mxbai");
    std::string experiment_name = (argc > 2) ? std::string(argv[2]) : std::string("end_to_end");
    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, wiki, arxiv, sift, fmnist\n";
        return 1;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);
    std::string filename = bench_utils::get_data_path(dataset);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
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
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    faiss::IndexFlatL2 index(d);

    faiss::ClusteringParameters cp;
    cp.niter = n_iters;
    cp.verbose = false;
    cp.max_points_per_centroid = 999999; // We don't want to take samples
    cp.nredo = 1;
    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(),
        bench_utils::ANGULAR_DATASETS.end(),
        dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means for dataset: " << dataset << std::endl;
        cp.spherical = true;
    }

    faiss::Clustering clus(d, n_clusters, cp);

    bench_utils::TicToc timer;
    timer.Tic();
    clus.train(n, data.data(), index);
    timer.Toc();

    double construction_time_ms = timer.GetMilliseconds();
    int actual_iterations = static_cast<int>(clus.iteration_stats.size());
    double final_objective = clus.iteration_stats.back().obj;

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;
    std::cout << "Final objective: " << final_objective << std::endl;

    std::string gt_filename = bench_utils::get_ground_truth_path(dataset);
    std::string queries_filename = bench_utils::get_query_path(dataset);
    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file(queries_filename, std::ios::binary);
    if (gt_file.good() && queries_file.good()) {
        gt_file.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;
        std::cout << "Queries file: " << queries_filename << std::endl;
        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
        int n_queries = bench_utils::N_QUERIES;
        std::cout << "Using " << n_queries << " queries (loaded " << gt_map.size()
                  << " from ground truth)" << std::endl;

        std::vector<float> queries(n_queries * d);
        queries_file.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
        queries_file.close();

        std::vector<faiss::idx_t> assignments(n);
        std::vector<float> distances_to_centroids(n);
        const float* centroids = clus.centroids.data();
        faiss::IndexFlatL2 centroid_index(d);
        centroid_index.add(n_clusters, centroids);
        centroid_index.search(n, data.data(), 1, distances_to_centroids.data(), assignments.data());

        auto results_knn_10 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids, n_queries, n_clusters, d, 10
        );
        bench_utils::print_recall_results(results_knn_10, 10);
        auto results_knn_100 = bench_utils::compute_recall(
            gt_map, assignments, queries.data(), centroids, n_queries, n_clusters, d, 100
        );
        bench_utils::print_recall_results(results_knn_100, 100);

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
    } else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file.good()) {
            std::cout << "Queries file not found: " << queries_filename << std::endl;
        }
        std::cout << "Skipping CSV output (recall computation requires ground truth)" << std::endl;
    }
}
