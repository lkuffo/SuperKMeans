#undef HAS_FFTW

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

int main() {
    constexpr size_t N_SAMPLES = 10000;
    constexpr size_t MAX_D = 768;
    constexpr size_t N_TRUE_CENTERS = 100;
    constexpr float CLUSTER_STD = 0.5f;
    constexpr float CENTER_SPREAD = 50.0f;
    constexpr unsigned int SEED = 42;
    constexpr int N_ITERS = 10;

    std::vector<size_t> k_values = {10, 100, 250};
    std::vector<size_t> d_values = {4, 16, 32, 64, 100, 128, 384, 512, 600, 768};

    omp_set_num_threads(1);

    std::cerr << "Generating test data (" << N_SAMPLES << " × " << MAX_D << ")...\n";
    auto full_data = skmeans::MakeBlobs(
        N_SAMPLES, MAX_D, N_TRUE_CENTERS, false, CLUSTER_STD, CENTER_SPREAD, SEED
    );

    std::string data_file = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    std::ofstream out(data_file, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Could not open " << data_file << " for writing\n";
        return 1;
    }
    out.write(reinterpret_cast<const char*>(full_data.data()), full_data.size() * sizeof(float));
    out.close();
    std::cerr << "Saved test data to " << data_file << " (" << (full_data.size() * sizeof(float) / 1024 / 1024) << " MB)\n\n";

    auto extract_subdim = [&](size_t d) -> std::vector<float> {
        if (d > MAX_D) {
            std::cerr << "Error: Requested d=" << d << " > MAX_D=" << MAX_D << "\n";
            return {};
        }
        std::vector<float> data(N_SAMPLES * d);
        for (size_t i = 0; i < N_SAMPLES; ++i) {
            std::memcpy(&data[i * d], &full_data[i * MAX_D], d * sizeof(float));
        }
        return data;
    };

    std::cout << std::scientific << std::setprecision(5);
    std::cout << "// Ground truth WCSS values for test_wcss.cpp\n";
    std::cout << "// Generated with: N_SAMPLES=" << N_SAMPLES << ", MAX_D=" << MAX_D
              << ", N_TRUE_CENTERS=" << N_TRUE_CENTERS << ", CLUSTER_STD=" << CLUSTER_STD
              << ", CENTER_SPREAD=" << CENTER_SPREAD << ", SEED=" << SEED << ", N_ITERS=" << N_ITERS
              << "\n";
    std::cout << "// Test data stored in: tests/test_data.bin\n";
    std::cout << "// Copy-paste the following into GROUND_TRUTH map:\n\n";

    for (size_t k : k_values) {
        std::cout << "    // k=" << k << "\n";
        for (size_t d : d_values) {

            auto data = extract_subdim(d);

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
