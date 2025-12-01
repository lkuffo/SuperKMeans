#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <faiss/utils/utils.h>

#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "superkmeans/nanobench.h"
#include "bench_utils.h"

int main(int argc, char* argv[]) {
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("openai");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
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
    std::string path_root = std::string(CMAKE_SOURCE_DIR) + "/benchmarks";
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
    std::string gt_filename = path_root + "/" + dataset + ".json";
    std::string queries_filename = path_root + "/data_" + dataset + "_test.bin";

    std::ifstream gt_file(gt_filename);
    std::ifstream queries_file(queries_filename, std::ios::binary);

    if (gt_file.good() && queries_file.good()) {
        gt_file.close();
        std::cout << "\n--- Computing Recall ---" << std::endl;
        std::cout << "Ground truth file: " << gt_filename << std::endl;
        std::cout << "Queries file: " << queries_filename << std::endl;

        // Load ground truth
        auto gt_map = bench_utils::parse_ground_truth_json(gt_filename);
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

        // Compute recall for different KNN values
        for (int knn : bench_utils::KNN_VALUES) {
            auto results = bench_utils::compute_recall(
                gt_map, assignments, queries.data(), centroids,
                n_queries, n_clusters, d, knn
            );
            bench_utils::print_recall_results(results, knn);
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
