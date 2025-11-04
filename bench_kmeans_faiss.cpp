#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include "superkmeans/nanobench.h"

int main(int argc, char* argv[]) {
    omp_set_num_threads(1);

    // SKMeans
    const int n = 262144;
    const int d = 1024;
    int n_clusters = 1024;
    int n_iters = 1;
    float sampling_fraction = 1.0;

    std::vector<float> data(n * d);
    std::ifstream file(std::string{CMAKE_SOURCE_DIR} + "/data_random.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    faiss::IndexFlatL2 index(d);

    // Set up clustering parameters
    faiss::ClusteringParameters cp;
    cp.niter = n_iters;        // number of k-means iterations
    cp.verbose = true;    // print progress

    // Create the clustering object
    faiss::Clustering clus(d, n_clusters, cp);

    // Perform clustering
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("FAISS KMeans", [&]() {
        clus.train(n, data.data(), index);
    });
    // Print centroids
    std::cout << "Centroids:" << std::endl;
}
