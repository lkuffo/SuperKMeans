#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/pdx/utils.h"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [n] [d] [k]\n"
              << "  n: Number of vectors (default: 1000000)\n"
              << "  d: Dimensionality (default: 768)\n"
              << "  k: Number of clusters (default: 1000)\n"
              << "\nExample:\n"
              << "  " << program_name << " 500000 512 100\n";
}

int main(int argc, char* argv[]) {
    size_t n = 1000000;
    size_t d = 768;
    size_t k = 10000;

    if (argc > 1) {
        if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        n = std::atoll(argv[1]);
    }
    if (argc > 2) {
        d = std::atoll(argv[2]);
    }
    if (argc > 3) {
        k = std::atoll(argv[3]);
    }

    std::cout << "Parameters: n=" << n << ", d=" << d << ", k=" << k << std::endl;
    std::cout << "Generating " << n << " vectors with d=" << d << std::endl;
    std::vector<float> data = skmeans::MakeBlobs(n, d, 100, true);

    auto kmeans = skmeans::HierarchicalSuperKMeans(k, d);

    // Run the clustering
    std::cout << "Running SuperKMeans with " << k << " clusters..." << std::endl;
    skmeans::TicToc timer;
    timer.Tic();
    std::vector<float> centroids = kmeans.Train(data.data(), n);
    timer.Toc();

    double construction_time_ms = timer.GetMilliseconds();
    std::cout << "Index built in: " << construction_time_ms << " ms" << std::endl;

    // Get assignments
    std::vector<uint32_t> assignments = kmeans.Assign(data.data(), centroids.data(), n, k);
}