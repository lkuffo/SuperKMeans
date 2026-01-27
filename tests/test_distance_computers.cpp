#include <gtest/gtest.h>
#include <omp.h>
#include <cmath>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/distance_computers/scalar_computers.h"
#include "superkmeans/pdx/utils.h"

namespace {

/**
 * @brief Generate random vectors for testing
 */
std::vector<float> GenerateRandomVectors(size_t n, size_t d, unsigned int seed = 42) {
    std::vector<float> data(n * d);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& val : data) {
        val = dist(gen);
    }
    return data;
}

/**
 * @brief Compute squared L2 norms for row-major vectors
 */
std::vector<float> ComputeNorms(const float* data, size_t n, size_t d) {
    std::vector<float> norms(n);
    for (size_t i = 0; i < n; ++i) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            norm += data[i * d + j] * data[i * d + j];
        }
        norms[i] = norm;
    }
    return norms;
}

/**
 * @brief Find nearest neighbor using brute force (reference implementation)
 */
void FindNearestNeighborBruteForce(
    const float* x,
    const float* y,
    size_t n_x,
    size_t n_y,
    size_t d,
    uint32_t* out_knn,
    float* out_distances
) {
    for (size_t i = 0; i < n_x; ++i) {
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best_idx = 0;

        for (size_t j = 0; j < n_y; ++j) {
            float dist = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                float diff = x[i * d + k] - y[j * d + k];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<uint32_t>(j);
            }
        }
        out_knn[i] = best_idx;
        out_distances[i] = best_dist;
    }
}

class DistanceComputerTest : public ::testing::Test {
  protected:
    void SetUp() override { omp_set_num_threads(omp_get_max_threads()); }
};

// ============================================================================
// Test 7: DistanceComputer SIMD matches Scalar
// ============================================================================

/**
 * @brief Test that SIMD L2 distance computation matches scalar reference
 *
 * The SIMD implementation (DistanceComputer) should produce the same results
 * as the scalar reference implementation (ScalarComputer) within floating-point
 * tolerance.
 */
TEST_F(DistanceComputerTest, SIMD_MatchesScalar_L2) {
    // Test various dimensionalities including non-aligned sizes
    std::vector<size_t> dimensions = {1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 
                                       100, 127, 128, 255, 256, 384, 512, 
                                       768, 1000, 1024, 1536, 2048};
    const size_t n_pairs = 100; // Number of vector pairs to test

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d));

        // Generate random vector pairs
        auto vectors1 = GenerateRandomVectors(n_pairs, d, 42);
        auto vectors2 = GenerateRandomVectors(n_pairs, d, 123);

        for (size_t i = 0; i < n_pairs; ++i) {
            const float* v1 = vectors1.data() + i * d;
            const float* v2 = vectors2.data() + i * d;

            // Compute with scalar reference
            float scalar_dist = skmeans::ScalarComputer<
                skmeans::DistanceFunction::l2, 
                skmeans::Quantization::f32
            >::Horizontal(v1, v2, d);

            // Compute with SIMD implementation
            float simd_dist = skmeans::DistanceComputer<
                skmeans::DistanceFunction::l2,
                skmeans::Quantization::f32
            >::Horizontal(v1, v2, d);

            // Check that results match within tolerance
            float abs_error = std::abs(scalar_dist - simd_dist);
            float rel_error = abs_error / std::max(scalar_dist, 1e-6f);

            EXPECT_LT(rel_error, 1e-5f)
                << "SIMD/Scalar mismatch at d=" << d << ", pair " << i
                << ": scalar=" << scalar_dist << ", simd=" << simd_dist
                << ", rel_error=" << rel_error;
        }
    }
}

/**
 * @brief Test SIMD distance with aligned memory
 */
TEST_F(DistanceComputerTest, SIMD_MatchesScalar_AlignedMemory) {
    const size_t d = 512;
    const size_t n_pairs = 50;

    // Allocate aligned memory
    alignas(64) float vectors1[n_pairs * d];
    alignas(64) float vectors2[n_pairs * d];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < n_pairs * d; ++i) {
        vectors1[i] = dist(gen);
        vectors2[i] = dist(gen);
    }

    for (size_t i = 0; i < n_pairs; ++i) {
        const float* v1 = vectors1 + i * d;
        const float* v2 = vectors2 + i * d;

        float scalar_dist = skmeans::ScalarComputer<
            skmeans::DistanceFunction::l2,
            skmeans::Quantization::f32
        >::Horizontal(v1, v2, d);

        float simd_dist = skmeans::DistanceComputer<
            skmeans::DistanceFunction::l2,
            skmeans::Quantization::f32
        >::Horizontal(v1, v2, d);

        float rel_error = std::abs(scalar_dist - simd_dist) / std::max(scalar_dist, 1e-6f);
        EXPECT_LT(rel_error, 1e-5f);
    }
}

/**
 * @brief Test that distance is zero for identical vectors
 */
TEST_F(DistanceComputerTest, SIMD_ZeroDistanceForIdenticalVectors) {
    std::vector<size_t> dimensions = {64, 128, 256, 512, 1024};

    for (size_t d : dimensions) {
        auto vectors = GenerateRandomVectors(10, d, 42);

        for (size_t i = 0; i < 10; ++i) {
            const float* v = vectors.data() + i * d;

            float dist = skmeans::DistanceComputer<
                skmeans::DistanceFunction::l2,
                skmeans::Quantization::f32
            >::Horizontal(v, v, d);

            EXPECT_NEAR(dist, 0.0f, 1e-10f)
                << "Distance to self should be zero at d=" << d;
        }
    }
}

/**
 * @brief Test triangle inequality: dist(a,c) <= dist(a,b) + dist(b,c)
 */
TEST_F(DistanceComputerTest, SIMD_TriangleInequality) {
    const size_t d = 256;
    const size_t n = 50;

    auto vectors = GenerateRandomVectors(n, d, 42);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            for (size_t k = j + 1; k < n; ++k) {
                const float* a = vectors.data() + i * d;
                const float* b = vectors.data() + j * d;
                const float* c = vectors.data() + k * d;

                float dist_ab = std::sqrt(skmeans::DistanceComputer<
                    skmeans::DistanceFunction::l2, skmeans::Quantization::f32
                >::Horizontal(a, b, d));

                float dist_bc = std::sqrt(skmeans::DistanceComputer<
                    skmeans::DistanceFunction::l2, skmeans::Quantization::f32
                >::Horizontal(b, c, d));

                float dist_ac = std::sqrt(skmeans::DistanceComputer<
                    skmeans::DistanceFunction::l2, skmeans::Quantization::f32
                >::Horizontal(a, c, d));

                // Triangle inequality (with small tolerance for floating point)
                EXPECT_LE(dist_ac, dist_ab + dist_bc + 1e-4f)
                    << "Triangle inequality violated for vectors " << i << "," << j << "," << k;
            }
        }
    }
}

// ============================================================================
// Test 8: BatchComputer FindNearestNeighbor Correctness
// ============================================================================

/**
 * @brief Test that BatchComputer::FindNearestNeighbor finds the true nearest neighbor
 *
 * Compares BLAS-based batch nearest neighbor search against brute force reference.
 */
TEST_F(DistanceComputerTest, BatchComputer_FindNearestNeighbor_Correctness) {
    // Test with various sizes
    struct TestCase {
        size_t n_x;
        size_t n_y;
        size_t d;
    };

    std::vector<TestCase> test_cases = {
        {100, 50, 64},
        {500, 100, 128},
        {1000, 200, 256},
        {2000, 500, 512},
        {5000, 1000, 128},
    };

    for (const auto& tc : test_cases) {
        SCOPED_TRACE("Testing n_x=" + std::to_string(tc.n_x) + 
                     ", n_y=" + std::to_string(tc.n_y) + 
                     ", d=" + std::to_string(tc.d));

        // Generate random data
        auto x = GenerateRandomVectors(tc.n_x, tc.d, 42);
        auto y = GenerateRandomVectors(tc.n_y, tc.d, 123);

        // Compute norms
        auto norms_x = ComputeNorms(x.data(), tc.n_x, tc.d);
        auto norms_y = ComputeNorms(y.data(), tc.n_y, tc.d);

        // Brute force reference
        std::vector<uint32_t> bf_knn(tc.n_x);
        std::vector<float> bf_distances(tc.n_x);
        FindNearestNeighborBruteForce(
            x.data(), y.data(), tc.n_x, tc.n_y, tc.d,
            bf_knn.data(), bf_distances.data()
        );

        // BatchComputer implementation
        std::vector<uint32_t> batch_knn(tc.n_x);
        std::vector<float> batch_distances(tc.n_x);
        std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);

        skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
            FindNearestNeighbor(
                x.data(), y.data(),
                tc.n_x, tc.n_y, tc.d,
                norms_x.data(), norms_y.data(),
                batch_knn.data(), batch_distances.data(),
                tmp_buf.data()
            );

        // Verify results
        size_t mismatches = 0;
        for (size_t i = 0; i < tc.n_x; ++i) {
            if (batch_knn[i] != bf_knn[i]) {
                // Check if distances are the same (tie-breaking)
                float rel_diff = std::abs(batch_distances[i] - bf_distances[i]) / 
                                std::max(bf_distances[i], 1e-6f);
                if (rel_diff > 1e-4f) {
                    ++mismatches;
                }
            }
        }

        EXPECT_EQ(mismatches, 0)
            << "Found " << mismatches << " mismatches in nearest neighbor results";

        // Verify distances match
        for (size_t i = 0; i < tc.n_x; ++i) {
            float rel_error = std::abs(batch_distances[i] - bf_distances[i]) / 
                             std::max(bf_distances[i], 1e-6f);
            EXPECT_LT(rel_error, 1e-4f)
                << "Distance mismatch at index " << i 
                << ": batch=" << batch_distances[i] << ", bf=" << bf_distances[i];
        }
    }
}

/**
 * @brief Test BatchComputer with synthetic clusterable data
 */
TEST_F(DistanceComputerTest, BatchComputer_FindNearestNeighbor_ClusterableData) {
    const size_t n_x = 1000;
    const size_t n_y = 100;  // Centroids
    const size_t d = 256;

    // Generate clusterable data
    auto x = skmeans::make_blobs(n_x, d, n_y, false, 1.0f, 10.0f, 42);
    auto y = GenerateRandomVectors(n_y, d, 123);

    auto norms_x = ComputeNorms(x.data(), n_x, d);
    auto norms_y = ComputeNorms(y.data(), n_y, d);

    // Brute force
    std::vector<uint32_t> bf_knn(n_x);
    std::vector<float> bf_distances(n_x);
    FindNearestNeighborBruteForce(x.data(), y.data(), n_x, n_y, d, bf_knn.data(), bf_distances.data());

    // BatchComputer
    std::vector<uint32_t> batch_knn(n_x);
    std::vector<float> batch_distances(n_x);
    std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);

    skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
        FindNearestNeighbor(
            x.data(), y.data(), n_x, n_y, d,
            norms_x.data(), norms_y.data(),
            batch_knn.data(), batch_distances.data(),
            tmp_buf.data()
        );

    // All indices should match (or have same distance for ties)
    for (size_t i = 0; i < n_x; ++i) {
        if (batch_knn[i] != bf_knn[i]) {
            // Verify it's a tie
            float rel_diff = std::abs(batch_distances[i] - bf_distances[i]) / 
                            std::max(bf_distances[i], 1e-6f);
            EXPECT_LT(rel_diff, 1e-4f)
                << "Index mismatch without distance tie at " << i;
        }
    }
}

/**
 * @brief Test BatchComputer with edge case: single query
 */
TEST_F(DistanceComputerTest, BatchComputer_SingleQuery) {
    const size_t n_y = 100;
    const size_t d = 128;

    auto x = GenerateRandomVectors(1, d, 42);
    auto y = GenerateRandomVectors(n_y, d, 123);

    auto norms_x = ComputeNorms(x.data(), 1, d);
    auto norms_y = ComputeNorms(y.data(), n_y, d);

    // Brute force
    std::vector<uint32_t> bf_knn(1);
    std::vector<float> bf_distances(1);
    FindNearestNeighborBruteForce(x.data(), y.data(), 1, n_y, d, bf_knn.data(), bf_distances.data());

    // BatchComputer
    std::vector<uint32_t> batch_knn(1);
    std::vector<float> batch_distances(1);
    std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);

    skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
        FindNearestNeighbor(
            x.data(), y.data(), 1, n_y, d,
            norms_x.data(), norms_y.data(),
            batch_knn.data(), batch_distances.data(),
            tmp_buf.data()
        );

    EXPECT_EQ(batch_knn[0], bf_knn[0]);
    float rel_error = std::abs(batch_distances[0] - bf_distances[0]) / std::max(bf_distances[0], 1e-6f);
    EXPECT_LT(rel_error, 1e-4f);
}

/**
 * @brief Test BatchComputer with high-dimensional data
 */
TEST_F(DistanceComputerTest, BatchComputer_HighDimensional) {
    const size_t n_x = 500;
    const size_t n_y = 100;
    const size_t d = 1536;  // High dimensional

    auto x = GenerateRandomVectors(n_x, d, 42);
    auto y = GenerateRandomVectors(n_y, d, 123);

    auto norms_x = ComputeNorms(x.data(), n_x, d);
    auto norms_y = ComputeNorms(y.data(), n_y, d);

    // Brute force
    std::vector<uint32_t> bf_knn(n_x);
    std::vector<float> bf_distances(n_x);
    FindNearestNeighborBruteForce(x.data(), y.data(), n_x, n_y, d, bf_knn.data(), bf_distances.data());

    // BatchComputer
    std::vector<uint32_t> batch_knn(n_x);
    std::vector<float> batch_distances(n_x);
    std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);

    skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
        FindNearestNeighbor(
            x.data(), y.data(), n_x, n_y, d,
            norms_x.data(), norms_y.data(),
            batch_knn.data(), batch_distances.data(),
            tmp_buf.data()
        );

    size_t mismatches = 0;
    for (size_t i = 0; i < n_x; ++i) {
        if (batch_knn[i] != bf_knn[i]) {
            float rel_diff = std::abs(batch_distances[i] - bf_distances[i]) / 
                            std::max(bf_distances[i], 1e-6f);
            if (rel_diff > 1e-4f) {
                ++mismatches;
            }
        }
    }

    EXPECT_EQ(mismatches, 0) << "Found " << mismatches << " mismatches in high-dim test";
}

} // anonymous namespace

