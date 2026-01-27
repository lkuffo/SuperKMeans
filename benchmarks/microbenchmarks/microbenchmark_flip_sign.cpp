#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/scalar_computers.h"
#include "superkmeans/pdx/utils.h"

constexpr size_t N_DIMENSIONS = 1024;
constexpr size_t N_ITERATIONS = 50000;
constexpr float FLIP_PROBABILITY = 0.5f; // 50% of values get sign flipped

/**
 * @brief Generates random float data and masks
 */
void GenerateRandomData(float* data, uint32_t* masks, size_t n_dimensions, float flip_probability) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility

    std::uniform_real_distribution<float> value_dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> flip_dist(0.0f, 1.0f);

    for (size_t i = 0; i < n_dimensions; ++i) {
        data[i] = value_dist(gen);
        // Set mask to 0x80000000 (flip) or 0 (keep)
        masks[i] = (flip_dist(gen) < flip_probability) ? 0x80000000 : 0;
    }
}

/**
 * @brief Benchmark scalar implementation
 */
double BenchmarkScalar(
    const float* data,
    const uint32_t* masks,
    size_t n_iterations,
    uint32_t& checksum_out
) {
    alignas(64) float buffer[N_DIMENSIONS];

    // Initialize buffer with input data
    std::memcpy(buffer, data, N_DIMENSIONS * sizeof(float));

    uint32_t checksum = 0;

    skmeans::TicToc timer;
    timer.Tic();

    for (size_t iter = 0; iter < n_iterations; ++iter) {
        // Flip in-place, creating a dependency chain
        skmeans::ScalarUtilsComputer<skmeans::Quantization::f32>::FlipSign(
            buffer, buffer, masks, N_DIMENSIONS
        );
        // Accumulate checksum to prevent optimization
        const uint32_t* buffer_bits = reinterpret_cast<const uint32_t*>(buffer);
        checksum ^= buffer_bits[iter % N_DIMENSIONS];
    }

    timer.Toc();

    // Store result to prevent optimization
    checksum_out = checksum;

    double avg_time_ns = static_cast<double>(timer.accum_time) / n_iterations;

    return avg_time_ns;
}

/**
 * @brief Benchmark SIMD implementation
 */
double BenchmarkSIMD(
    const float* data,
    const uint32_t* masks,
    size_t n_iterations,
    uint32_t& checksum_out
) {
    alignas(64) float buffer[N_DIMENSIONS];

    // Initialize buffer with input data
    std::memcpy(buffer, data, N_DIMENSIONS * sizeof(float));

    uint32_t checksum = 0;

    skmeans::TicToc timer;
    timer.Tic();

    for (size_t iter = 0; iter < n_iterations; ++iter) {
        // Flip in-place, creating a dependency chain
        skmeans::UtilsComputer<skmeans::Quantization::f32>::FlipSign(
            buffer, buffer, masks, N_DIMENSIONS
        );
        // Accumulate checksum to prevent optimization
        const uint32_t* buffer_bits = reinterpret_cast<const uint32_t*>(buffer);
        checksum ^= buffer_bits[iter % N_DIMENSIONS];
    }

    timer.Toc();

    // Store result to prevent optimization
    checksum_out = checksum;

    double avg_time_ns = static_cast<double>(timer.accum_time) / n_iterations;

    return avg_time_ns;
}

/**
 * @brief Verify both implementations produce the same results
 */
bool VerifyCorrectness(const float* data, const uint32_t* masks) {
    alignas(64) float scalar_output[N_DIMENSIONS];
    alignas(64) float simd_output[N_DIMENSIONS];

    // Run scalar version
    skmeans::ScalarUtilsComputer<skmeans::Quantization::f32>::FlipSign(
        data, scalar_output, masks, N_DIMENSIONS
    );

    // Run SIMD version
    skmeans::UtilsComputer<skmeans::Quantization::f32>::FlipSign(
        data, simd_output, masks, N_DIMENSIONS
    );

    // Compare outputs (bitwise comparison for exact match)
    for (size_t i = 0; i < N_DIMENSIONS; ++i) {
        uint32_t scalar_bits = *reinterpret_cast<const uint32_t*>(&scalar_output[i]);
        uint32_t simd_bits = *reinterpret_cast<const uint32_t*>(&simd_output[i]);

        if (scalar_bits != simd_bits) {
            std::cerr << "ERROR: Output mismatch at index " << i << "! Scalar: " << scalar_output[i]
                      << " (0x" << std::hex << scalar_bits << ")"
                      << ", SIMD: " << simd_output[i] << " (0x" << simd_bits << ")" << std::dec
                      << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    std::cout << "=== FlipSign Microbenchmark ===" << std::endl;
    std::cout << "n_dimensions: " << N_DIMENSIONS << std::endl;
    std::cout << "n_iterations: " << N_ITERATIONS << std::endl;
    std::cout << "flip_probability: " << (FLIP_PROBABILITY * 100.0f) << "%" << std::endl;
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
    alignas(64) float data[N_DIMENSIONS];
    alignas(64) uint32_t masks[N_DIMENSIONS];

    GenerateRandomData(data, masks, N_DIMENSIONS, FLIP_PROBABILITY);

    // Verify correctness first
    std::cout << "Verifying correctness... ";
    if (!VerifyCorrectness(data, masks)) {
        std::cerr << "FAILED!" << std::endl;
        return 1;
    }
    std::cout << "PASSED" << std::endl;
    std::cout << std::endl;

    // Warm up
    std::cout << "Warming up... " << std::flush;
    uint32_t warmup_checksum;
    BenchmarkScalar(data, masks, 1000, warmup_checksum);
    BenchmarkSIMD(data, masks, 1000, warmup_checksum);
    std::cout << "done" << std::endl;
    std::cout << std::endl;

    // Benchmark
    std::cout << "Running benchmarks..." << std::endl;

    uint32_t scalar_checksum = 0;
    uint32_t simd_checksum = 0;

    double scalar_time_ns = BenchmarkScalar(data, masks, N_ITERATIONS, scalar_checksum);
    double simd_time_ns = BenchmarkSIMD(data, masks, N_ITERATIONS, simd_checksum);

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

    // Calculate throughput
    double scalar_gb_per_sec =
        (N_DIMENSIONS * sizeof(float) * 2) / scalar_time_ns; // 2 for read+write
    double simd_gb_per_sec = (N_DIMENSIONS * sizeof(float) * 2) / simd_time_ns;

    std::cout << std::endl;
    std::cout << "Throughput:" << std::endl;
    std::cout << "Scalar:  " << std::setprecision(2) << scalar_gb_per_sec << " GB/s" << std::endl;
    std::cout << "SIMD:    " << std::setprecision(2) << simd_gb_per_sec << " GB/s" << std::endl;

    // Sanity check: print checksums
    std::cout << std::endl;
    std::cout << "Checksums (verification that work was done):" << std::endl;
    std::cout << "Scalar:  0x" << std::hex << scalar_checksum << std::dec << std::endl;
    std::cout << "SIMD:    0x" << std::hex << simd_checksum << std::dec << std::endl;

    return 0;
}
