/**
 * Test to verify that Assign() produces the same results as the internal assignments from Train().
 * 
 * When sampling_fraction = 1.0 (no sampling), the assignments computed during Train() (using rotated data
 * and rotated centroids) should match the assignments from Assign() (using raw data and unrotated centroids).
 * 
 * This test validates:
 * 1. The Assign() function works correctly
 * 2. The Unrotate() function correctly transforms centroids back to original space
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>

#include "superkmeans/common.h"
#include "superkmeans/superkmeans.h"

int main(int argc, char* argv[]) {
    // Choose dataset by name
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("sift");

    const std::unordered_map<std::string, std::pair<size_t, size_t>> dataset_params = {
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
        std::cerr << "Known datasets: mxbai, openai, arxiv, sift, fmnist, glove100, glove50\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters = std::max<size_t>(1u, static_cast<size_t>(std::sqrt(static_cast<double>(n)) * 4.0));
    const int n_iters = 25;
    const float sampling_fraction = 1.0f;  // No sampling - use all data
    
    std::string path_root = std::string(CMAKE_SOURCE_DIR);
    std::string filename = path_root + "/data_" + dataset + ".bin";
    
    constexpr size_t THREADS = 10;
    omp_set_num_threads(THREADS);

    std::cout << "=== Assign() Verification Test ===\n";
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters=" << n_clusters << ", n_iters=" << n_iters << "\n";
    std::cout << "sampling_fraction=" << sampling_fraction << " (using all data)\n\n";

    // Load data
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> data;
    try {
        data.resize(n * d);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate data vector: " << e.what() << "\n";
        return 1;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();
    std::cout << "Loaded " << n << " vectors from " << filename << "\n\n";

    // Create and train k-means
    std::cout << "Running Train()...\n";
    auto kmeans = skmeans::SuperKMeans<skmeans::f32, skmeans::l2>(
        n_clusters, d, n_iters, sampling_fraction, true, THREADS
    );
    
    // Train and get unrotated centroids
    // Enable perform_assignments=true so _assignments matches the returned centroids
    auto centroids = kmeans.Train(
        data.data(), n,
        nullptr,  // queries
        0,        // n_queries
        false,    // sample_queries
        100,      // objective_k
        0.01f,    // ann_explore_fraction
        true,     // unrotate_centroids
        true      // perform_assignments - do final assignment pass
    );
    
    // Get the internal assignments from Train()
    const auto& train_assignments = kmeans._assignments;
    std::cout << "\nTrain() completed. Got " << train_assignments.size() << " assignments.\n\n";

    // Now run Assign() with raw data and unrotated centroids
    std::cout << "Running Assign() with raw data and unrotated centroids...\n";
    auto assign_assignments = skmeans::SuperKMeans<skmeans::f32, skmeans::l2>::Assign(
        data.data(),
        centroids.data(),
        n,
        n_clusters,
        d
    );
    std::cout << "Assign() completed. Got " << assign_assignments.size() << " assignments.\n\n";

    // Compare assignments
    std::cout << "Comparing assignments...\n";
    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (train_assignments[i] != assign_assignments[i]) {
            ++mismatches;
            if (mismatches <= 10) {
                std::cout << "  Mismatch at index " << i << ": Train=" << train_assignments[i] 
                          << ", Assign=" << assign_assignments[i] << "\n";
            }
        }
    }

    double mismatch_pct = 100.0 * static_cast<double>(mismatches) / static_cast<double>(n);
    std::cout << "\n=== Results ===\n";
    std::cout << "Total vectors: " << n << "\n";
    std::cout << "Mismatches: " << mismatches << " (" << mismatch_pct << "%)\n";

    // Check if within tolerance (0.01%)
    const double tolerance = 0.01;
    if (mismatch_pct <= tolerance) {
        std::cout << "\n✓ TEST PASSED: Mismatch rate (" << mismatch_pct << "%) is within tolerance (" << tolerance << "%)\n";
        std::cout << "  This confirms that Unrotate() and Assign() work correctly!\n";
        return 0;
    } else {
        std::cout << "\n✗ TEST FAILED: Mismatch rate (" << mismatch_pct << "%) exceeds tolerance (" << tolerance << "%)\n";
        std::cout << "  This suggests an issue with Unrotate() or Assign() implementation.\n";
        return 1;
    }
}

