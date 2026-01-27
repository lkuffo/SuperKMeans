/**
 * @file test_rotation.cpp
 * @brief Tests for ADSamplingPruner rotation functionality
 *
 * Tests cover:
 * - Rotate/Unrotate inverse property
 * - Norm preservation (orthogonal transformation property)
 * - Inner product preservation (orthogonal transformation property)
 * - Both orthonormal matrix (d < 512) and DCT (d >= 512) rotation methods
 */

#include <cmath>
#include <gtest/gtest.h>
#include <omp.h>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/adsampling.h"

namespace {

/**
 * @brief Generate random vectors for testing
 */
std::vector<float> GenerateRandomVectors(size_t n, size_t d, unsigned int seed = 42) {
    std::vector<float> output(n * d);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : output) {
        val = dist(rng);
    }
    return output;
}

/**
 * @brief Compute L2 norm of a vector
 */
float ComputeNorm(const float* v, size_t d) {
    float norm_sq = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        norm_sq += v[i] * v[i];
    }
    return std::sqrt(norm_sq);
}

/**
 * @brief Compute inner product of two vectors
 */
float ComputeInnerProduct(const float* a, const float* b, size_t d) {
    float dot = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}

/**
 * @brief Get rotation method name for a given dimension
 */
std::string GetRotationMethod(size_t d) {
    // Note: On x86 with AVX2, DCT requires power-of-2 dimensions
#ifdef __AVX2__
    bool uses_dct = (d >= skmeans::D_THRESHOLD_FOR_DCT_ROTATION) && 
                    skmeans::IsPowerOf2(static_cast<uint32_t>(d));
#else
    bool uses_dct = (d >= skmeans::D_THRESHOLD_FOR_DCT_ROTATION);
#endif
    return uses_dct ? "DCT" : "Orthonormal Matrix";
}

class RotationTest : public ::testing::Test {
  protected:
    void SetUp() override { omp_set_num_threads(omp_get_max_threads()); }
};

// ============================================================================
// Rotate/Unrotate Inverse Property Tests
// ============================================================================

/**
 * @brief Test that Rotate followed by Unrotate returns original vectors (low dim)
 */
TEST_F(RotationTest, RotateUnrotateInverse_LowDim) {
    const size_t d = 128;
    const size_t n = 100;

    auto original = GenerateRandomVectors(n, d, 42);

    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

    std::vector<float> rotated(n * d);
    pruner.Rotate(original.data(), rotated.data(), n);

    std::vector<float> recovered(n * d);
    pruner.Unrotate(rotated.data(), recovered.data(), n);

    double max_error = 0.0;
    double sum_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        double error = std::abs(original[i] - recovered[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }
    double avg_error = sum_error / (n * d);

    EXPECT_LT(max_error, 1e-4) << "Max error too large for d=" << d;
    EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
}

/**
 * @brief Test that Rotate followed by Unrotate returns original vectors (high dim, DCT)
 */
TEST_F(RotationTest, RotateUnrotateInverse_HighDim_DCT) {
    const size_t d = 1024;
    const size_t n = 100;

    ASSERT_GE(d, skmeans::D_THRESHOLD_FOR_DCT_ROTATION)
        << "Test expects d >= D_THRESHOLD_FOR_DCT_ROTATION to use DCT rotation";

    auto original = GenerateRandomVectors(n, d, 42);

    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

    std::vector<float> rotated(n * d);
    pruner.Rotate(original.data(), rotated.data(), n);

    std::vector<float> recovered(n * d);
    pruner.Unrotate(rotated.data(), recovered.data(), n);

    double max_error = 0.0;
    double sum_error = 0.0;
    for (size_t i = 0; i < n * d; ++i) {
        double error = std::abs(original[i] - recovered[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }
    double avg_error = sum_error / (n * d);

    EXPECT_LT(max_error, 1e-4) << "Max error too large for d=" << d;
    EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
}

/**
 * @brief Test Rotate/Unrotate inverse across multiple dimensions
 */
TEST_F(RotationTest, RotateUnrotateInverse_MultipleDimensions) {
    std::vector<size_t> dimensions = {50, 128, 256, 512, 768, 1024, 2048};
    const size_t n = 100;

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d) + " (" + GetRotationMethod(d) + ")");

        auto original = GenerateRandomVectors(n, d, 42);

        skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(original.data(), rotated.data(), n);

        std::vector<float> recovered(n * d);
        pruner.Unrotate(rotated.data(), recovered.data(), n);

        double max_error = 0.0;
        double sum_error = 0.0;
        for (size_t i = 0; i < n * d; ++i) {
            double error = std::abs(original[i] - recovered[i]);
            max_error = std::max(max_error, error);
            sum_error += error;
        }
        double avg_error = sum_error / (n * d);

        EXPECT_LT(max_error, 1e-4)
            << "Max error too large for d=" << d << " (" << GetRotationMethod(d) << ")";
        EXPECT_LT(avg_error, 1e-5) << "Average error too large for d=" << d;
    }
}

// ============================================================================
// Test 12: Both DCT and Orthonormal preserve norms
// ============================================================================

/**
 * @brief Test that both rotation methods preserve vector L2 norms
 *
 * An orthogonal transformation (rotation) preserves the L2 norm of vectors.
 * This is a fundamental property that ensures distances are preserved.
 */
TEST_F(RotationTest, BothRotationMethodsPreserveNorms) {
    // Test dimensions that use orthonormal matrix (d < 512)
    std::vector<size_t> orthonormal_dims = {64, 128, 256, 384};
    // Test dimensions that use DCT (d >= 512, power of 2 on x86)
    std::vector<size_t> dct_dims = {512, 1024, 2048};

    const size_t n = 50;

    // Test orthonormal matrix rotation
    for (size_t d : orthonormal_dims) {
        SCOPED_TRACE("Testing orthonormal rotation d=" + std::to_string(d));

        auto original = GenerateRandomVectors(n, d, 42);
        skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(original.data(), rotated.data(), n);

        for (size_t i = 0; i < n; ++i) {
            float original_norm = ComputeNorm(original.data() + i * d, d);
            float rotated_norm = ComputeNorm(rotated.data() + i * d, d);

            float rel_error = std::abs(original_norm - rotated_norm) / original_norm;
            EXPECT_LT(rel_error, 1e-5f)
                << "Orthonormal rotation should preserve norm for vector " << i
                << " at d=" << d << " (original=" << original_norm 
                << ", rotated=" << rotated_norm << ")";
        }
    }

    // Test DCT rotation
    for (size_t d : dct_dims) {
        SCOPED_TRACE("Testing DCT rotation d=" + std::to_string(d));

        auto original = GenerateRandomVectors(n, d, 123);
        skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(original.data(), rotated.data(), n);

        for (size_t i = 0; i < n; ++i) {
            float original_norm = ComputeNorm(original.data() + i * d, d);
            float rotated_norm = ComputeNorm(rotated.data() + i * d, d);

            float rel_error = std::abs(original_norm - rotated_norm) / original_norm;
            EXPECT_LT(rel_error, 1e-5f)
                << "DCT rotation should preserve norm for vector " << i << " at d=" << d
                << " (original=" << original_norm << ", rotated=" << rotated_norm << ")";
        }
    }
}

// ============================================================================
// Test 10: Rotation preserves inner products
// ============================================================================

/**
 * @brief Test that rotation preserves inner products between vectors
 *
 * For an orthogonal transformation Q: <Qx, Qy> = <x, y>
 * This is essential because L2 distance is derived from inner products:
 * ||x - y||² = ||x||² + ||y||² - 2<x, y>
 *
 * If inner products are preserved, distances are preserved.
 */
TEST_F(RotationTest, RotationPreservesInnerProducts) {
    std::vector<size_t> dimensions = {64, 128, 256, 512, 1024};
    const size_t n = 30;  // Test n*(n-1)/2 pairs

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d) + " (" + GetRotationMethod(d) + ")");

        auto vectors = GenerateRandomVectors(n, d, 42);

        skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(vectors.data(), rotated.data(), n);

        // Check inner products for all pairs
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const float* vi_orig = vectors.data() + i * d;
                const float* vj_orig = vectors.data() + j * d;
                const float* vi_rot = rotated.data() + i * d;
                const float* vj_rot = rotated.data() + j * d;

                float orig_dot = ComputeInnerProduct(vi_orig, vj_orig, d);
                float rot_dot = ComputeInnerProduct(vi_rot, vj_rot, d);

                float abs_error = std::abs(orig_dot - rot_dot);
                
                // For very small dot products, use absolute error threshold
                // For larger dot products, use relative error threshold
                // Both should be satisfied for orthogonal transformations within numerical precision
                bool abs_ok = abs_error < 1e-4f;
                float rel_error = abs_error / std::max(std::abs(orig_dot), 1.0f);
                bool rel_ok = rel_error < 1e-3f;

                EXPECT_TRUE(abs_ok || rel_ok)
                    << "Inner product not preserved for vectors " << i << " and " << j
                    << " at d=" << d << " (" << GetRotationMethod(d) << ")"
                    << ": original=" << orig_dot << ", rotated=" << rot_dot
                    << " (abs_err=" << abs_error << ", rel_err=" << rel_error << ")";
            }
        }
    }
}

/**
 * @brief Test that rotation preserves L2 distances (derived from inner products)
 */
TEST_F(RotationTest, RotationPreservesDistances) {
    std::vector<size_t> dimensions = {128, 512, 1024};
    const size_t n = 20;

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d) + " (" + GetRotationMethod(d) + ")");

        auto vectors = GenerateRandomVectors(n, d, 42);

        skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

        std::vector<float> rotated(n * d);
        pruner.Rotate(vectors.data(), rotated.data(), n);

        // Check distances for all pairs
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const float* vi_orig = vectors.data() + i * d;
                const float* vj_orig = vectors.data() + j * d;
                const float* vi_rot = rotated.data() + i * d;
                const float* vj_rot = rotated.data() + j * d;

                // Compute L2 distance squared
                float orig_dist_sq = 0.0f;
                float rot_dist_sq = 0.0f;
                for (size_t k = 0; k < d; ++k) {
                    float orig_diff = vi_orig[k] - vj_orig[k];
                    float rot_diff = vi_rot[k] - vj_rot[k];
                    orig_dist_sq += orig_diff * orig_diff;
                    rot_dist_sq += rot_diff * rot_diff;
                }

                float rel_error = std::abs(orig_dist_sq - rot_dist_sq) / orig_dist_sq;

                EXPECT_LT(rel_error, 1e-4f)
                    << "Distance not preserved for vectors " << i << " and " << j
                    << " at d=" << d << " (" << GetRotationMethod(d) << ")"
                    << ": original=" << std::sqrt(orig_dist_sq) 
                    << ", rotated=" << std::sqrt(rot_dist_sq);
            }
        }
    }
}

// ============================================================================
// Edge cases and additional tests
// ============================================================================

/**
 * @brief Test rotation with a single vector
 */
TEST_F(RotationTest, SingleVector) {
    std::vector<size_t> dimensions = {64, 512, 1024};

    for (size_t d : dimensions) {
        auto original = GenerateRandomVectors(1, d, 42);

        skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(d, 2.1f);

        std::vector<float> rotated(d);
        pruner.Rotate(original.data(), rotated.data(), 1);

        std::vector<float> recovered(d);
        pruner.Unrotate(rotated.data(), recovered.data(), 1);

        float max_error = 0.0f;
        for (size_t i = 0; i < d; ++i) {
            max_error = std::max(max_error, std::abs(original[i] - recovered[i]));
        }

        EXPECT_LT(max_error, 1e-4f) << "Single vector test failed for d=" << d;
    }
}

/**
 * @brief Test that different seeds produce different rotations
 */
TEST_F(RotationTest, DifferentSeedsProduceDifferentRotations) {
    const size_t d = 256;
    const size_t n = 10;

    auto original = GenerateRandomVectors(n, d, 42);

    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner1(d, 2.1f, 42);
    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner2(d, 2.1f, 123);

    std::vector<float> rotated1(n * d);
    std::vector<float> rotated2(n * d);

    pruner1.Rotate(original.data(), rotated1.data(), n);
    pruner2.Rotate(original.data(), rotated2.data(), n);

    // Rotated vectors should be different
    bool found_difference = false;
    for (size_t i = 0; i < n * d; ++i) {
        if (std::abs(rotated1[i] - rotated2[i]) > 1e-6f) {
            found_difference = true;
            break;
        }
    }

    EXPECT_TRUE(found_difference)
        << "Different seeds should produce different rotation matrices";
}

/**
 * @brief Test that same seed produces identical rotations
 */
TEST_F(RotationTest, SameSeedProducesIdenticalRotations) {
    const size_t d = 256;
    const size_t n = 10;

    auto original = GenerateRandomVectors(n, d, 42);

    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner1(d, 2.1f, 42);
    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner2(d, 2.1f, 42);

    std::vector<float> rotated1(n * d);
    std::vector<float> rotated2(n * d);

    pruner1.Rotate(original.data(), rotated1.data(), n);
    pruner2.Rotate(original.data(), rotated2.data(), n);

    // Rotated vectors should be identical
    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_FLOAT_EQ(rotated1[i], rotated2[i])
            << "Same seed should produce identical rotations at index " << i;
    }
}

} // anonymous namespace

