#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/scalar_computers.h"
#include "superkmeans/pdx/utils.h"

constexpr size_t N_ITERATIONS = 10000;
constexpr size_t DIMENSIONALITIES[] = {64, 128, 256, 512, 768, 1024, 1536};
constexpr size_t N_DIMS = sizeof(DIMENSIONALITIES) / sizeof(DIMENSIONALITIES[0]);

/**
 * @brief Benchmark scalar Horizontal implementation
 */
double BenchmarkScalarHorizontal(
    const float* vector1,
    const float* vector2,
    size_t n_dimensions,
    size_t n_iterations,
    float& sum_out
) {
    float sum = 0.0f;

    skmeans::TicToc timer;
    timer.Tic();
    for (size_t iter = 0; iter < n_iterations; ++iter) {
        float distance =
            skmeans::ScalarComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
                Horizontal(vector1, vector2, n_dimensions);
        sum += distance;
    }
    timer.Toc();
    sum_out = sum;
    double avg_time_ns = static_cast<double>(timer.accum_time) / n_iterations;
    return avg_time_ns;
}

/**
 * @brief Benchmark SIMD Horizontal implementation
 */
double BenchmarkSIMDHorizontal(
    const float* vector1,
    const float* vector2,
    size_t n_dimensions,
    size_t n_iterations,
    float& sum_out
) {
    float sum = 0.0f;
    skmeans::TicToc timer;
    timer.Tic();
    for (size_t iter = 0; iter < n_iterations; ++iter) {
        float distance =
            skmeans::DistanceComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
                Horizontal(vector1, vector2, n_dimensions);
        sum += distance;
    }
    timer.Toc();
    sum_out = sum;
    double avg_time_ns = static_cast<double>(timer.accum_time) / n_iterations;
    return avg_time_ns;
}

/**
 * @brief Verify both implementations produce the same results
 */
bool VerifyCorrectness(const float* vector1, const float* vector2, size_t n_dimensions) {
    float scalar_result =
        skmeans::ScalarComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
            Horizontal(vector1, vector2, n_dimensions);
    float simd_result =
        skmeans::DistanceComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
            Horizontal(vector1, vector2, n_dimensions);
    float abs_error = std::abs(scalar_result - simd_result);
    float rel_error = abs_error / std::max(std::abs(scalar_result), 1e-6f);
    if (rel_error > 1e-4f) {
        std::cerr << "ERROR: Result mismatch for d=" << n_dimensions
                  << "! Scalar: " << scalar_result << ", SIMD: " << simd_result
                  << ", Relative error: " << rel_error << std::endl;
        return false;
    }
    return true;
}

struct BenchmarkResult {
    size_t dimensionality;
    double scalar_time_ns;
    double simd_time_ns;
    double speedup;
};

int main() {
    std::cout << "=== Horizontal L2 Distance Microbenchmark ===" << std::endl;
    std::cout << "n_iterations per test: " << N_ITERATIONS << std::endl;
    std::cout << std::endl;
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

    constexpr size_t MAX_DIM = 1536;
    alignas(64) float vector1[MAX_DIM];
    alignas(64) float vector2[MAX_DIM];

    std::vector<BenchmarkResult> results;
    std::cout << "Verifying correctness for all dimensionalities... ";
    for (size_t dim_idx = 0; dim_idx < N_DIMS; ++dim_idx) {
        size_t d = DIMENSIONALITIES[dim_idx];
        auto v1 = skmeans::GenerateRandomVectors(1, d, -10.0f, 10.0f, 42);
        auto v2 = skmeans::GenerateRandomVectors(1, d, -10.0f, 10.0f, 123);
        std::copy(v1.begin(), v1.end(), vector1);
        std::copy(v2.begin(), v2.end(), vector2);
        if (!VerifyCorrectness(vector1, vector2, d)) {
            std::cerr << "FAILED at d=" << d << "!" << std::endl;
            return 1;
        }
    }
    std::cout << "PASSED" << std::endl;
    std::cout << std::endl;

    std::cout << "Running benchmarks..." << std::endl;
    std::cout << std::endl;
    for (size_t dim_idx = 0; dim_idx < N_DIMS; ++dim_idx) {
        size_t d = DIMENSIONALITIES[dim_idx];
        std::cout << "Testing d=" << d << "... " << std::flush;

        auto v1 = skmeans::GenerateRandomVectors(1, d, -10.0f, 10.0f, 42);
        auto v2 = skmeans::GenerateRandomVectors(1, d, -10.0f, 10.0f, 123);
        std::copy(v1.begin(), v1.end(), vector1);
        std::copy(v2.begin(), v2.end(), vector2);
        float warmup_sum;
        BenchmarkScalarHorizontal(vector1, vector2, d, 100, warmup_sum);
        BenchmarkSIMDHorizontal(vector1, vector2, d, 100, warmup_sum);
        float scalar_sum = 0.0f;
        float simd_sum = 0.0f;
        double scalar_time =
            BenchmarkScalarHorizontal(vector1, vector2, d, N_ITERATIONS, scalar_sum);
        double simd_time = BenchmarkSIMDHorizontal(vector1, vector2, d, N_ITERATIONS, simd_sum);

        BenchmarkResult result;
        result.dimensionality = d;
        result.scalar_time_ns = scalar_time;
        result.simd_time_ns = simd_time;
        result.speedup = scalar_time / simd_time;
        results.push_back(result);
        std::cout << "done" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(8) << "Dim" << std::setw(15) << "Scalar (ns)" << std::setw(15)
              << "SIMD (ns)" << std::setw(12) << "Speedup" << std::setw(18) << "Improvement (%)"
              << std::endl;
    std::cout << std::string(68, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    for (const auto& result : results) {
        double improvement = (result.speedup - 1.0) * 100.0;
        std::cout << std::setw(8) << result.dimensionality << std::setw(15) << result.scalar_time_ns
                  << std::setw(15) << result.simd_time_ns << std::setw(12) << result.speedup << "x"
                  << std::setw(16) << improvement << "%" << std::endl;
    }

    std::cout << std::endl;
    double avg_speedup = 0.0;
    for (const auto& result : results) {
        avg_speedup += result.speedup;
    }
    avg_speedup /= results.size();
    std::cout << "Average speedup: " << std::setprecision(2) << avg_speedup << "x" << std::endl;
    auto best = std::max_element(
        results.begin(),
        results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) { return a.speedup < b.speedup; }
    );
    auto worst = std::min_element(
        results.begin(),
        results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) { return a.speedup < b.speedup; }
    );
    std::cout << "Best speedup:    " << std::setprecision(2) << best->speedup
              << "x (d=" << best->dimensionality << ")" << std::endl;
    std::cout << "Worst speedup:   " << std::setprecision(2) << worst->speedup
              << "x (d=" << worst->dimensionality << ")" << std::endl;
    return 0;
}
