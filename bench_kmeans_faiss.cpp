#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "superkmeans/nanobench.h"

int main(int argc, char* argv[]) {
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("glove100");

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
    int n_iters = 25;
    float sampling_fraction = 1.0;
    constexpr size_t THREADS = 14;
    omp_set_num_threads(THREADS);
    std::string path_root = std::string(CMAKE_SOURCE_DIR);
    std::string filename = path_root + "/data_" + dataset + ".bin";

    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";

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
    // Print centroids
    std::cout << "Centroids:" << std::endl;
}
