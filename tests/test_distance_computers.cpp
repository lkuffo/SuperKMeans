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

class DistanceComputerTest : public ::testing::Test {
  protected:
    void SetUp() override { omp_set_num_threads(omp_get_max_threads()); }
};

/**
 * @brief Test that SIMD L2 distance computation matches scalar reference
 *
 */
TEST_F(DistanceComputerTest, SIMD_MatchesScalar_L2) {
    std::vector<size_t> dimensions = {1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 
                                       100, 127, 128, 255, 256, 384, 512, 
                                       768, 1000, 1024, 1536, 2048};
    const size_t n_pairs = 100;

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d));

        auto vectors1 = skmeans::GenerateRandomVectors(n_pairs, d, -10.0f, 10.0f, 42);
        auto vectors2 = skmeans::GenerateRandomVectors(n_pairs, d, -10.0f, 10.0f, 123);

        for (size_t i = 0; i < n_pairs; ++i) {
            const float* v1 = vectors1.data() + i * d;
            const float* v2 = vectors2.data() + i * d;

            float scalar_dist = skmeans::ScalarComputer<
                skmeans::DistanceFunction::l2, 
                skmeans::Quantization::f32
            >::Horizontal(v1, v2, d);

            float simd_dist = skmeans::DistanceComputer<
                skmeans::DistanceFunction::l2,
                skmeans::Quantization::f32
            >::Horizontal(v1, v2, d);

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
 * @brief Test that distance is zero for identical vectors
 */
TEST_F(DistanceComputerTest, SIMD_ZeroDistanceForIdenticalVectors) {
    std::vector<size_t> dimensions = {64, 128, 256, 512, 1024};

    for (size_t d : dimensions) {
        auto vectors = skmeans::GenerateRandomVectors(10, d, -10.0f, 10.0f, 42);

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
 * @brief Test that BatchComputer::FindNearestNeighbor finds the true nearest neighbor
 *
 * Compares BLAS-based batch nearest neighbor search against brute force reference.
 */
TEST_F(DistanceComputerTest, BatchComputer_FindNearestNeighbor_Correctness) {
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

        auto x = skmeans::GenerateRandomVectors(tc.n_x, tc.d, -10.0f, 10.0f, 42);
        auto y = skmeans::GenerateRandomVectors(tc.n_y, tc.d, -10.0f, 10.0f, 123);
        auto norms_x = skmeans::ComputeNorms(x.data(), tc.n_x, tc.d);
        auto norms_y = skmeans::ComputeNorms(y.data(), tc.n_y, tc.d);

        std::vector<uint32_t> bf_knn(tc.n_x);
        std::vector<float> bf_distances(tc.n_x);
        skmeans::FindNearestNeighborBruteForce(
            x.data(), y.data(), tc.n_x, tc.n_y, tc.d,
            bf_knn.data(), bf_distances.data()
        );

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
 * @brief Test BatchComputer with edge case: single query
 */
TEST_F(DistanceComputerTest, BatchComputer_SingleQuery) {
    const size_t n_y = 100;
    const size_t d = 128;

    auto x = skmeans::GenerateRandomVectors(1, d, -10.0f, 10.0f, 42);
    auto y = skmeans::GenerateRandomVectors(n_y, d, -10.0f, 10.0f, 123);

    auto norms_x = skmeans::ComputeNorms(x.data(), 1, d);
    auto norms_y = skmeans::ComputeNorms(y.data(), n_y, d);

    std::vector<uint32_t> bf_knn(1);
    std::vector<float> bf_distances(1);
    skmeans::FindNearestNeighborBruteForce(x.data(), y.data(), 1, n_y, d, bf_knn.data(), bf_distances.data());

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
 * @brief Test that BatchComputer::FindKNearestNeighbors finds the true k nearest neighbors
 *
 * Compares BLAS-based batch k-NN search against brute force reference with k=10.
 */
TEST_F(DistanceComputerTest, BatchComputer_FindKNearestNeighbors_Correctness) {
    const size_t n_x = 1000;
    const size_t n_y = 500;
    const size_t d = 128;
    const size_t k = 10;

    auto x = skmeans::GenerateRandomVectors(n_x, d, -10.0f, 10.0f, 42);
    auto y = skmeans::GenerateRandomVectors(n_y, d, -10.0f, 10.0f, 123);
    auto norms_x = skmeans::ComputeNorms(x.data(), n_x, d);
    auto norms_y = skmeans::ComputeNorms(y.data(), n_y, d);

    std::vector<uint32_t> bf_knn(n_x * k);
    std::vector<float> bf_distances(n_x * k);
    skmeans::FindKNearestNeighborsBruteForce(
        x.data(), y.data(), n_x, n_y, d, k,
        bf_knn.data(), bf_distances.data()
    );

    std::vector<uint32_t> batch_knn(n_x * k);
    std::vector<float> batch_distances(n_x * k);
    std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);
    skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>::
        FindKNearestNeighbors(
            x.data(), y.data(),
            n_x, n_y, d,
            norms_x.data(), norms_y.data(),
            k,
            batch_knn.data(), batch_distances.data(),
            tmp_buf.data()
        );

    size_t index_mismatches = 0;
    size_t distance_mismatches = 0;
    for (size_t i = 0; i < n_x; ++i) {
        for (size_t ki = 0; ki < k; ++ki) {
            size_t idx = i * k + ki;
            // Check if indices match
            if (batch_knn[idx] != bf_knn[idx]) {
                // Allow mismatch if distances are equal (tie-breaking)
                float rel_diff = std::abs(batch_distances[idx] - bf_distances[idx]) / 
                                std::max(bf_distances[idx], 1e-6f);
                if (rel_diff > 1e-4f) {
                    ++index_mismatches;
                }
            }
            // Check distance accuracy
            float rel_error = std::abs(batch_distances[idx] - bf_distances[idx]) / 
                             std::max(bf_distances[idx], 1e-6f);
            if (rel_error > 1e-4f) {
                ++distance_mismatches;
            }
        }
    }

    EXPECT_EQ(index_mismatches, 0)
        << "Found " << index_mismatches << " index mismatches in k-NN results";
    EXPECT_EQ(distance_mismatches, 0)
        << "Found " << distance_mismatches << " distance mismatches in k-NN results";

    for (size_t i = 0; i < n_x; ++i) {
        for (size_t ki = 1; ki < k; ++ki) {
            EXPECT_LE(batch_distances[i * k + ki - 1], batch_distances[i * k + ki])
                << "Results not sorted for query " << i << " at position " << ki;
        }
    }
}

} // anonymous namespace

