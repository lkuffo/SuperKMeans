#undef HAS_FFTW

#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#include "recall_utils.h"

int main() {
    constexpr size_t N_SAMPLES = 10000;
    constexpr size_t MAX_D = 1024;
    constexpr unsigned int SEED = 42;
    constexpr int N_ITERS = 10;

    std::vector<size_t> k_values = {10, 100, 250};
    std::vector<size_t> d_values = {4, 16, 32, 64, 100, 128, 384, 512, 600, 768, 1024};

    omp_set_num_threads(1);

    std::string data_file = CMAKE_SOURCE_DIR "/tests/test_data.bin";

    std::cout << std::scientific << std::setprecision(5);
    std::cout << "// Ground truth WCSS values for test_wcss.cpp (data: tests/test_data.bin)\n";
    std::cout << "// Copy-paste the following into GROUND_TRUTH map:\n\n";

    for (size_t k : k_values) {
        std::cout << "    // k=" << k << "\n";
        for (size_t d : d_values) {
            auto data = skm_test::LoadTestDataSubdim(data_file, N_SAMPLES, MAX_D, d);

            // These config values MUST match those in test_wcss.cpp
            skmeans::SuperKMeansConfig config;
            config.iters = N_ITERS;
            config.verbose = false;
            config.seed = SEED;
            config.early_termination = false;
            config.sampling_fraction = 1.0f;
            config.max_points_per_cluster = 99999;
            config.min_not_pruned_pct = 0.03f;
            config.max_not_pruned_pct = 0.05f;
            config.adjustment_factor_for_partial_d = 0.20f;
            config.angular = false;
            config.n_threads = 1;

            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    k, d, config
                );
            kmeans.Train(data.data(), N_SAMPLES);
            float wcss = kmeans.iteration_stats.back().objective;
            std::cout << "    {{" << k << ", " << d << "}, " << wcss << "f},\n";
        }
    }
    return 0;
}
