#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <faiss/utils/utils.h>

#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <algorithm>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "superkmeans/nanobench.h"

// Simple JSON parser for our specific use case
std::unordered_map<int, std::vector<int>> parse_ground_truth_json(const std::string& filename) {
    std::unordered_map<int, std::vector<int>> gt_map;
    std::ifstream file(filename);
    if (!file.is_open()) {
        return gt_map;
    }

    std::string line;
    std::getline(file, line); // Read entire file as one line (it's a single JSON object)

    // Simple parser: look for "query_idx": [vector_ids...]
    size_t pos = 0;
    while ((pos = line.find("\"", pos)) != std::string::npos) {
        size_t key_start = pos + 1;
        size_t key_end = line.find("\"", key_start);
        if (key_end == std::string::npos) break;

        std::string key_str = line.substr(key_start, key_end - key_start);
        int query_idx = std::stoi(key_str);

        // Find the array of vector IDs
        size_t arr_start = line.find("[", key_end);
        size_t arr_end = line.find("]", arr_start);
        if (arr_start == std::string::npos || arr_end == std::string::npos) break;

        std::string arr_str = line.substr(arr_start + 1, arr_end - arr_start - 1);
        std::vector<int> vector_ids;
        std::istringstream iss(arr_str);
        std::string token;
        while (std::getline(iss, token, ',')) {
            // Remove whitespace
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) {
                vector_ids.push_back(std::stoi(token));
            }
        }

        gt_map[query_idx] = vector_ids;
        pos = arr_end + 1;
    }

    return gt_map;
}

int main(int argc, char* argv[]) {
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("mxbai");

    const std::unordered_map<std::string, std::pair<int, int>> dataset_params = {
        {"mxbai", {769382, 1024}},
        {"openai", {999000, 1536}},
        {"arxiv", {2253000, 768}},
        {"sift", {1000000, 128}},
        {"fmnist", {60000, 784}},
        {"glove100", {1183514, 100}},
        {"glove50", {1183514, 50}}
    };

    auto it = dataset_params.find(dataset);
    if (it == dataset_params.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, arxiv, sift, fmnist\n";
        return 1;
    }

    const int n = it->second.first;
    const int d = it->second.second;
    const int n_clusters =
        std::max<int>(1u, static_cast<int>(std::sqrt(static_cast<double>(n)) * 4.0));
    int n_iters = 2;
    float sampling_fraction = 1.0;
    constexpr size_t THREADS = 14;
    omp_set_num_threads(THREADS);
    std::string path_root = std::string(CMAKE_SOURCE_DIR);
    std::string filename = path_root + "/data_" + dataset + ".bin";

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

    // Set up clustering parameters
    faiss::ClusteringParameters cp;
    cp.niter = n_iters; // number of k-means iterations
    cp.verbose = true;  // print progress
    // cp.max_points_per_centroid = 500;
    cp.nredo = 1;

    // Create the clustering object
    faiss::Clustering clus(d, n_clusters, cp);

    // Perform clustering
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("FAISS KMeans", [&]() {
        clus.train(n, data.data(), index);
    });
    std::cout << "Obj:" << clus.iteration_stats[n_iters - 1].obj << std::endl;

    // Compute recall if ground truth file exists
    std::string gt_filename = path_root + "/agnews-" + dataset + "-" + std::to_string(d) + "-euclidean_10.json";
    std::string queries_filename = path_root + "/data_" + dataset + "_test.bin";

    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file(queries_filename, std::ios::binary);

    if (gt_file.good() && queries_file.good()) {
        gt_file.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;
        std::cout << "Queries file: " << queries_filename << std::endl;

        // Load ground truth
        auto gt_map = parse_ground_truth_json(gt_filename);
        int n_queries = gt_map.size();
        std::cout << "Loaded " << n_queries << " queries from ground truth" << std::endl;

        // Load query vectors
        std::vector<float> queries(n_queries * d);
        queries_file.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
        queries_file.close();

        // Get cluster assignments from FAISS
        // FAISS doesn't store assignments directly, so we need to assign data points to nearest centroids
        std::vector<faiss::idx_t> assignments(n);
        std::vector<float> distances_to_centroids(n);

        // Get centroids from clustering result
        const float* centroids = clus.centroids.data();

        // Assign each data point to its nearest centroid
        faiss::IndexFlatL2 centroid_index(d);
        centroid_index.add(n_clusters, centroids);
        centroid_index.search(n, data.data(), 1, distances_to_centroids.data(), assignments.data());

        // Compute distances from queries to centroids
        // Using L2 distance: ||q - c||^2 = ||q||^2 + ||c||^2 - 2*q·c
        std::vector<float> query_norms(n_queries);
        std::vector<float> centroid_norms(n_clusters);

        // Compute query norms
        for (int i = 0; i < n_queries; ++i) {
            float norm = 0.0f;
            for (int j = 0; j < d; ++j) {
                float val = queries[i * d + j];
                norm += val * val;
            }
            query_norms[i] = norm;
        }

        // Compute centroid norms
        for (int i = 0; i < n_clusters; ++i) {
            float norm = 0.0f;
            for (int j = 0; j < d; ++j) {
                float val = centroids[i * d + j];
                norm += val * val;
            }
            centroid_norms[i] = norm;
        }

        // Compute query-centroid distances
        std::vector<float> distances(n_queries * n_clusters);
        for (int i = 0; i < n_queries; ++i) {
            for (int j = 0; j < n_clusters; ++j) {
                // Dot product
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += queries[i * d + k] * centroids[j * d + k];
                }
                distances[i * n_clusters + j] = query_norms[i] + centroid_norms[j] - 2.0f * dot;
            }
        }

        // Test different numbers of centroids to explore
        std::vector<float> explore_fractions = {
            0.001f, 0.002f, 0.003f, 0.004f, 0.005f, 0.006f, 0.007f, 0.008f, 0.009f,
            0.0100f, 0.0125f, 0.0150f, 0.0175f, 0.0200f, 0.0225f, 0.0250f, 0.0275f,
            0.0300f, 0.0325f, 0.0350f, 0.0375f, 0.0400f, 0.0425f, 0.0450f, 0.0475f, 0.0500f,
            0.1f
        };

        const int KNN = 10;  // Use first 10 ground truth vectors

        for (float explore_frac : explore_fractions) {
            int centroids_to_explore = std::max(1, static_cast<int>(n_clusters * explore_frac));

            // For each query, find top-N nearest centroids
            float total_recall = 0.0f;

            for (int query_idx = 0; query_idx < n_queries; ++query_idx) {
                if (gt_map.find(query_idx) == gt_map.end()) {
                    continue;
                }

                // Get distances for this query
                std::vector<std::pair<float, int>> query_distances;
                for (int j = 0; j < n_clusters; ++j) {
                    query_distances.push_back({distances[query_idx * n_clusters + j], j});
                }

                // Sort by distance to get top-N centroids
                std::partial_sort(query_distances.begin(),
                                query_distances.begin() + centroids_to_explore,
                                query_distances.end());

                // Create set of top centroid indices
                std::unordered_set<int> top_centroids;
                for (int t = 0; t < centroids_to_explore; ++t) {
                    top_centroids.insert(query_distances[t].second);
                }

                // Check how many ground truth vectors have their assigned centroid in top-N
                const auto& gt_vector_ids = gt_map[query_idx];
                int found = 0;
                int gt_count = std::min(KNN, static_cast<int>(gt_vector_ids.size()));

                for (int i = 0; i < gt_count; ++i) {
                    int vector_id = gt_vector_ids[i];
                    int assigned_centroid = assignments[vector_id];
                    if (top_centroids.find(assigned_centroid) != top_centroids.end()) {
                        ++found;
                    }
                }

                float query_recall = static_cast<float>(found) / static_cast<float>(gt_count);
                total_recall += query_recall;
            }

            float average_recall = total_recall / static_cast<float>(n_queries);
            printf("Recall@%4d (%5.2f%% of centroids): %.4f\n",
                   centroids_to_explore, explore_frac * 100.0f, average_recall);
        }
    } else {
        if (!gt_file.good()) {
            std::cout << "\nGround truth file not found: " << gt_filename << std::endl;
        }
        if (!queries_file.good()) {
            std::cout << "Queries file not found: " << queries_filename << std::endl;
        }
    }
}
