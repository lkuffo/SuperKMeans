/**
 * @file generate_wcss_ground_truth.cpp
 * @brief Generates ground truth WCSS values for test_wcss.cpp
 *
 * Run this program to regenerate the GROUND_TRUTH map in test_wcss.cpp
 * when parameters change or after algorithm improvements.
 *
 * Build: make generate_wcss_ground_truth.out
 * Run:   ./tests/generate_wcss_ground_truth.out > ground_truth_values.txt
 */

#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

int main() {
    // These parameters MUST match those in test_wcss.cpp
    constexpr size_t N_SAMPLES = 100000;
    constexpr size_t N_TRUE_CENTERS = 100;
    constexpr float CLUSTER_STD = 1.0f;
    constexpr float CENTER_SPREAD = 10.0f;
    constexpr unsigned int SEED = 42;
    constexpr int N_ITERS = 10;

    std::vector<size_t> k_values = {10, 100, 1000, 10000};
    std::vector<size_t> d_values = {4,   16,  32,   64,   100,  128,  384, 512,
                                    600, 768, 900, 1024, 1536, 2000, 3072};

    omp_set_num_threads(omp_get_max_threads());

    std::cout << std::scientific << std::setprecision(5);
    std::cout << "// Ground truth WCSS values for test_wcss.cpp\n";
    std::cout << "// Generated with: N_SAMPLES=" << N_SAMPLES
              << ", N_TRUE_CENTERS=" << N_TRUE_CENTERS << ", CLUSTER_STD=" << CLUSTER_STD
              << ", CENTER_SPREAD=" << CENTER_SPREAD << ", SEED=" << SEED
              << ", N_ITERS=" << N_ITERS << "\n";
    std::cout << "// Copy-paste the following into GROUND_TRUTH map:\n\n";

    for (size_t k : k_values) {
        std::cout << "    // k=" << k << "\n";
        for (size_t d : d_values) {
            std::cerr << "Running k=" << k << ", d=" << d << "..." << std::flush;

            auto data = skmeans::make_blobs(
                N_SAMPLES, d, N_TRUE_CENTERS, false, CLUSTER_STD, CENTER_SPREAD, SEED
            );

            // These config values MUST match those in test_wcss.cpp
            skmeans::SuperKMeansConfig config;
            config.iters = N_ITERS;
            config.verbose = false;
            config.seed = SEED;
            config.early_termination = false;
            config.sampling_fraction = 1.0f;
            // PDX pruning parameters (explicit for reproducibility)
            config.min_not_pruned_pct = 0.03f;
            config.max_not_pruned_pct = 0.05f;
            config.adjustment_factor_for_partial_d = 0.20f;
            config.angular = false;

            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    k, d, config
                );
            kmeans.Train(data.data(), N_SAMPLES);

            float wcss = kmeans.iteration_stats.back().objective;

            std::cout << "    {{" << k << ", " << d << "}, " << wcss << "f},\n";
            std::cerr << " WCSS=" << wcss << "\n";
        }
    }

    std::cout << "\n// End of ground truth values\n";
    return 0;
}
