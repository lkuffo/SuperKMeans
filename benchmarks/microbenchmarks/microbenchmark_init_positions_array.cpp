#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/scalar_computers.h"
#include "superkmeans/pdx/utils.h"

constexpr size_t N_VECTORS = 1024;
constexpr size_t N_ITERATIONS = 10000;
constexpr float SELECTIVITY = 0.03f; // 3% of values below threshold

/**
 * @brief Generates random distances where only SELECTIVITY% are below threshold
 */
void GenerateRandomDistances(
    float* pruning_distances,
    size_t n_vectors,
    float threshold,
    float selectivity
) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility

    // Calculate how many values should be below threshold
    size_t n_below = static_cast<size_t>(n_vectors * selectivity);

    // Generate values below threshold
    std::uniform_real_distribution<float> below_dist(0.0f, threshold * 0.99f);
    for (size_t i = 0; i < n_below; ++i) {
        pruning_distances[i] = below_dist(gen);
    }

    // Generate values above threshold
    std::uniform_real_distribution<float> above_dist(threshold * 1.01f, threshold * 10.0f);
    for (size_t i = n_below; i < n_vectors; ++i) {
        pruning_distances[i] = above_dist(gen);
    }

    // Shuffle to randomize positions
    std::shuffle(pruning_distances, pruning_distances + n_vectors, gen);
}

/**
 * @brief Benchmark scalar implementation
 */
double BenchmarkScalar(
    const float* pruning_distances,
    float pruning_threshold,
    size_t n_iterations,
    size_t& total_count_out,
    uint32_t& first_pos_out
) {
    alignas(64) uint32_t pruning_positions[N_VECTORS];
    size_t n_vectors_not_pruned = 0;
    size_t total_count = 0;

    skmeans::TicToc timer;
    timer.Tic();

    for (size_t iter = 0; iter < n_iterations; ++iter) {
        skmeans::ScalarUtilsComputer<skmeans::Quantization::f32>::InitPositionsArray(
            N_VECTORS, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances
        );
        total_count += n_vectors_not_pruned;
    }

    timer.Toc();

    // Store results to prevent optimization
    total_count_out = total_count;
    first_pos_out = pruning_positions[0];

    double avg_time_ns = static_cast<double>(timer.accum_time) / n_iterations;

    return avg_time_ns;
}

/**
 * @brief Benchmark SIMD implementation
 */
double BenchmarkSIMD(
    const float* pruning_distances,
    float pruning_threshold,
    size_t n_iterations,
    size_t& total_count_out,
    uint32_t& first_pos_out
) {
    alignas(64) uint32_t pruning_positions[N_VECTORS];
    size_t n_vectors_not_pruned = 0;
    size_t total_count = 0;

    skmeans::TicToc timer;
    timer.Tic();

    for (size_t iter = 0; iter < n_iterations; ++iter) {
        skmeans::UtilsComputer<skmeans::Quantization::f32>::InitPositionsArray(
            N_VECTORS, n_vectors_not_pruned, pruning_positions, pruning_threshold, pruning_distances
        );
        total_count += n_vectors_not_pruned;
    }

    timer.Toc();

    // Store results to prevent optimization
    total_count_out = total_count;
    first_pos_out = pruning_positions[0];

    double avg_time_ns = static_cast<double>(timer.accum_time) / n_iterations;

    return avg_time_ns;
}

/**
 * @brief Verify both implementations produce the same results
 */
bool VerifyCorrectness(const float* pruning_distances, float pruning_threshold) {
    alignas(64) uint32_t scalar_positions[N_VECTORS];
    alignas(64) uint32_t simd_positions[N_VECTORS];
    size_t scalar_count = 0;
    size_t simd_count = 0;

    // Run scalar version
    skmeans::ScalarUtilsComputer<skmeans::Quantization::f32>::InitPositionsArray(
        N_VECTORS, scalar_count, scalar_positions, pruning_threshold, pruning_distances
    );

    // Run SIMD version
    skmeans::UtilsComputer<skmeans::Quantization::f32>::InitPositionsArray(
        N_VECTORS, simd_count, simd_positions, pruning_threshold, pruning_distances
    );

    // Compare counts
    if (scalar_count != simd_count) {
        std::cerr << "ERROR: Count mismatch! Scalar: " << scalar_count << ", SIMD: " << simd_count
                  << std::endl;
        return false;
    }

    // Compare positions
    for (size_t i = 0; i < scalar_count; ++i) {
        if (scalar_positions[i] != simd_positions[i]) {
            std::cerr << "ERROR: Position mismatch at index " << i
                      << "! Scalar: " << scalar_positions[i] << ", SIMD: " << simd_positions[i]
                      << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    std::cout << "=== InitPositionsArray Microbenchmark ===" << std::endl;
    std::cout << "n_vectors: " << N_VECTORS << std::endl;
    std::cout << "n_iterations: " << N_ITERATIONS << std::endl;
    std::cout << "selectivity: " << (SELECTIVITY * 100.0f) << "%" << std::endl;
    std::cout << std::endl;

    // Determine SIMD architecture
    std::string simd_arch = "Unknown";
#if defined(__AVX512F__)
    simd_arch = "AVX-512";
#elif defined(__AVX2__)
    simd_arch = "AVX2";
#elif defined(__ARM_NEON)
    simd_arch = "NEON";
#else
    simd_arch = "Scalar (no SIMD)";
#endif
    std::cout << "SIMD Architecture: " << simd_arch << std::endl;
    std::cout << std::endl;

    // Setup test data
    alignas(64) float pruning_distances[N_VECTORS];
    const float pruning_threshold = 100.0f;

    GenerateRandomDistances(pruning_distances, N_VECTORS, pruning_threshold, SELECTIVITY);

    // Verify correctness first
    std::cout << "Verifying correctness... ";
    if (!VerifyCorrectness(pruning_distances, pruning_threshold)) {
        std::cerr << "FAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    std::cout << std::endl;

    // Warm up
    std::cout << "Warming up... " << std::flush;
    size_t warmup_count;
    uint32_t warmup_pos;
    BenchmarkScalar(pruning_distances, pruning_threshold, 1000, warmup_count, warmup_pos);
    BenchmarkSIMD(pruning_distances, pruning_threshold, 1000, warmup_count, warmup_pos);
    std::cout << "done" << std::endl;
    std::cout << std::endl;

    // Benchmark
    std::cout << "Running benchmarks..." << std::endl;

    size_t scalar_total_count = 0;
    size_t simd_total_count = 0;
    uint32_t scalar_first_pos = 0;
    uint32_t simd_first_pos = 0;

    double scalar_time_ns = BenchmarkScalar(
        pruning_distances, pruning_threshold, N_ITERATIONS, scalar_total_count, scalar_first_pos
    );
    double simd_time_ns = BenchmarkSIMD(
        pruning_distances, pruning_threshold, N_ITERATIONS, simd_total_count, simd_first_pos
    );

    // Print results
    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Scalar:  " << std::setw(8) << scalar_time_ns << " ns/call" << std::endl;
    std::cout << "SIMD:    " << std::setw(8) << simd_time_ns << " ns/call" << std::endl;
    std::cout << std::endl;

    double speedup = scalar_time_ns / simd_time_ns;
    std::cout << "Speedup: " << std::setprecision(2) << speedup << "x" << std::endl;

    if (speedup > 1.0) {
        std::cout << "SIMD is " << std::setprecision(1) << ((speedup - 1.0) * 100.0) << "% faster"
                  << std::endl;
    } else {
        std::cout << "Scalar is " << std::setprecision(1) << ((1.0 / speedup - 1.0) * 100.0)
                  << "% faster" << std::endl;
    }

    // Sanity check: both should have processed the same total count
    std::cout << std::endl;
    std::cout << "Avg elements passing threshold: " << (scalar_total_count / N_ITERATIONS)
              << std::endl;

    return 0;
}
