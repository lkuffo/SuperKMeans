/**
 * @file test_pdx_layout.cpp
 * @brief Tests for PDX layout dimension splitting and PDXify transformation
 */

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/layout.h"

namespace {

/**
 * @brief Generate random test data
 */
std::vector<float> GenerateTestData(size_t n, size_t d, unsigned int seed = 42) {
    std::vector<float> data(n * d);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : data) {
        v = dist(rng);
    }
    return data;
}

/**
 * @brief Access a value from PDX layout given original row-major indices
 *
 * PDX layout within a chunk:
 * - Vertical dimensions (first vertical_d): stored column-major
 * - Horizontal dimensions: stored in row-major blocks of H_DIM_SIZE
 *
 * @param pdx_data Pointer to PDX-formatted data
 * @param chunk_idx Which chunk (0-indexed)
 * @param vec_in_chunk Vector index within the chunk
 * @param dim_idx Dimension index
 * @param chunk_size Number of vectors per chunk
 * @param d Total dimensions
 * @param vertical_d Number of vertical dimensions
 * @return The value at the specified position
 */
template <size_t CHUNK_SIZE>
float AccessPDX(
    const float* pdx_data,
    size_t chunk_idx,
    size_t vec_in_chunk,
    size_t dim_idx,
    size_t d,
    size_t vertical_d,
    size_t actual_chunk_size  // For last chunk which may be smaller
) {
    size_t chunk_offset = chunk_idx * CHUNK_SIZE * d;

    if (dim_idx < vertical_d) {
        // Vertical: column-major within chunk
        return pdx_data[chunk_offset + dim_idx * actual_chunk_size + vec_in_chunk];
    } else {
        // Horizontal: row-major blocks of H_DIM_SIZE
        size_t h_dim = dim_idx - vertical_d;
        size_t block_idx = h_dim / skmeans::H_DIM_SIZE;
        size_t dim_in_block = h_dim % skmeans::H_DIM_SIZE;

        size_t block_start = chunk_offset + vertical_d * actual_chunk_size +
                             block_idx * actual_chunk_size * skmeans::H_DIM_SIZE;
        return pdx_data[block_start + vec_in_chunk * skmeans::H_DIM_SIZE + dim_in_block];
    }
}

/**
 * @brief Test 14: PDX dimension split correctness for various dimensions
 *
 * Verifies:
 * 1. horizontal_d is a multiple of H_DIM_SIZE (64) or 0
 * 2. horizontal_d + vertical_d = d
 */
TEST(PDXLayoutTest, DimensionSplit_CorrectForVariousDimensions) {
    // Test a wide range of dimensions including edge cases
    std::vector<size_t> dimensions = {
        // Very small (edge cases)
        1, 2, 4, 8, 16, 32, 48, 63, 64, 65,
        // Around H_DIM_SIZE boundaries
        127, 128, 129, 191, 192, 193, 255, 256, 257,
        // Common embedding dimensions
        384, 512, 600, 768, 900, 1024, 1536, 2048, 3072, 4096
    };

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d));

        auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);

        // Check 1: horizontal_d is a multiple of H_DIM_SIZE or 0
        EXPECT_TRUE(split.horizontal_d % skmeans::H_DIM_SIZE == 0 || split.horizontal_d == 0)
            << "horizontal_d=" << split.horizontal_d
            << " is not a multiple of " << skmeans::H_DIM_SIZE << " (or 0) for d=" << d;

        // Check 2: horizontal_d + vertical_d = d
        EXPECT_EQ(split.horizontal_d + split.vertical_d, d)
            << "horizontal_d=" << split.horizontal_d << " + vertical_d=" << split.vertical_d
            << " != d=" << d;
    }
}

/**
 * @brief Test that small dimensions (d <= H_DIM_SIZE) have horizontal_d = 0
 */
TEST(PDXLayoutTest, SmallDimensions_HorizontalIsZero) {
    for (size_t d = 1; d <= skmeans::H_DIM_SIZE; ++d) {
        auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);

        EXPECT_EQ(split.horizontal_d, 0u)
            << "For d=" << d << " <= H_DIM_SIZE, horizontal_d should be 0";
        EXPECT_EQ(split.vertical_d, d)
            << "For d=" << d << " <= H_DIM_SIZE, vertical_d should equal d";
    }
}

/**
 * @brief Test that vertical_d is never zero (always some vertical dimensions)
 */
TEST(PDXLayoutTest, VerticalNeverZero) {
    std::vector<size_t> dimensions = {64, 128, 256, 512, 768, 1024, 2048, 4096};

    for (size_t d : dimensions) {
        auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);

        EXPECT_GT(split.vertical_d, 0u)
            << "vertical_d should never be zero for d=" << d;
    }
}

// ============================================================================
// PDXify Correctness Tests
// ============================================================================

/**
 * @brief Test PDXify preserves all data values (no data loss)
 *
 * After PDXify, we should be able to retrieve every original value
 * by accessing the correct position in the PDX layout.
 */
TEST(PDXLayoutTest, PDXify_PreservesAllValues) {
    // Use a small chunk size for testing
    constexpr size_t TEST_CHUNK_SIZE = 64;

    // Test various dimensions that have horizontal components
    std::vector<size_t> dimensions = {128, 256, 384, 512};
    const size_t n = TEST_CHUNK_SIZE * 2 + 17;  // 2 full chunks + partial

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing PDXify d=" + std::to_string(d) + ", n=" + std::to_string(n));

        auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);

        // Skip if no horizontal dimensions (PDXify requires horizontal_d % H_DIM_SIZE == 0)
        if (split.horizontal_d == 0) continue;

        auto input = GenerateTestData(n, d, 42);
        std::vector<float> pdx_output(n * d);

        // Apply PDXify with custom chunk size
        skmeans::PDXLayout<>::PDXify<false, TEST_CHUNK_SIZE>(
            input.data(), pdx_output.data(), n, d
        );

        // Verify we can retrieve every original value
        size_t full_chunks = n / TEST_CHUNK_SIZE;
        size_t remaining = n % TEST_CHUNK_SIZE;

        // Check full chunks
        for (size_t chunk = 0; chunk < full_chunks; ++chunk) {
            for (size_t vec = 0; vec < TEST_CHUNK_SIZE; ++vec) {
                for (size_t dim = 0; dim < d; ++dim) {
                    size_t orig_idx = (chunk * TEST_CHUNK_SIZE + vec) * d + dim;
                    float expected = input[orig_idx];
                    float actual = AccessPDX<TEST_CHUNK_SIZE>(
                        pdx_output.data(), chunk, vec, dim, d, split.vertical_d, TEST_CHUNK_SIZE
                    );

                    EXPECT_FLOAT_EQ(expected, actual)
                        << "Mismatch at chunk=" << chunk << ", vec=" << vec << ", dim=" << dim
                        << " (vertical_d=" << split.vertical_d << ")";
                }
            }
        }

        // Check partial chunk
        if (remaining > 0) {
            for (size_t vec = 0; vec < remaining; ++vec) {
                for (size_t dim = 0; dim < d; ++dim) {
                    size_t orig_idx = (full_chunks * TEST_CHUNK_SIZE + vec) * d + dim;
                    float expected = input[orig_idx];
                    float actual = AccessPDX<TEST_CHUNK_SIZE>(
                        pdx_output.data(), full_chunks, vec, dim, d, split.vertical_d, remaining
                    );

                    EXPECT_FLOAT_EQ(expected, actual)
                        << "Mismatch in partial chunk at vec=" << vec << ", dim=" << dim;
                }
            }
        }
    }
}

/**
 * @brief Test PDXify output size matches input size
 */
TEST(PDXLayoutTest, PDXify_OutputSizeMatchesInput) {
    constexpr size_t TEST_CHUNK_SIZE = 64;
    std::vector<size_t> dimensions = {128, 256, 512};
    std::vector<size_t> sizes = {100, 200, TEST_CHUNK_SIZE * 3};

    for (size_t d : dimensions) {
        auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);
        if (split.horizontal_d == 0) continue;

        for (size_t n : sizes) {
            auto input = GenerateTestData(n, d, 42);
            std::vector<float> output(n * d, -999.0f);  // Fill with sentinel

            skmeans::PDXLayout<>::PDXify<false, TEST_CHUNK_SIZE>(
                input.data(), output.data(), n, d
            );

            // Count how many values were written (not sentinel)
            size_t written = std::count_if(output.begin(), output.end(),
                [](float v) { return v != -999.0f; });

            EXPECT_EQ(written, n * d)
                << "PDXify should write exactly n*d values for n=" << n << ", d=" << d;
        }
    }
}

/**
 * @brief Test PDXify vertical dimensions are column-major within chunk
 */
TEST(PDXLayoutTest, PDXify_VerticalIsColumnMajor) {
    constexpr size_t TEST_CHUNK_SIZE = 8;
    const size_t d = 128;
    const size_t n = TEST_CHUNK_SIZE;  // Single full chunk

    auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);

    // Create input where each element is uniquely identifiable
    std::vector<float> input(n * d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            input[i * d + j] = static_cast<float>(i * 1000 + j);  // vec_idx * 1000 + dim_idx
        }
    }

    std::vector<float> pdx_output(n * d);
    skmeans::PDXLayout<>::PDXify<false, TEST_CHUNK_SIZE>(
        input.data(), pdx_output.data(), n, d
    );

    // Check vertical dimensions are stored column-major
    // In column-major, consecutive memory addresses differ by vector index, not dimension
    for (size_t dim = 0; dim < split.vertical_d; ++dim) {
        for (size_t vec = 0; vec < n; ++vec) {
            size_t pdx_idx = dim * n + vec;  // Column-major indexing
            float expected = static_cast<float>(vec * 1000 + dim);
            EXPECT_FLOAT_EQ(pdx_output[pdx_idx], expected)
                << "Vertical dim " << dim << ", vec " << vec << " not in column-major order";
        }
    }
}

/**
 * @brief Test PDXify horizontal dimensions are in row-major H_DIM_SIZE blocks
 */
TEST(PDXLayoutTest, PDXify_HorizontalIsRowMajorBlocks) {
    constexpr size_t TEST_CHUNK_SIZE = 8;
    const size_t d = 256;  // Needs horizontal dimensions
    const size_t n = TEST_CHUNK_SIZE;

    auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);
    ASSERT_GT(split.horizontal_d, 0u) << "Test requires horizontal dimensions";

    // Create input where each element is uniquely identifiable
    std::vector<float> input(n * d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            input[i * d + j] = static_cast<float>(i * 10000 + j);
        }
    }

    std::vector<float> pdx_output(n * d);
    skmeans::PDXLayout<>::PDXify<false, TEST_CHUNK_SIZE>(
        input.data(), pdx_output.data(), n, d
    );

    // After vertical section, horizontal blocks start
    size_t horizontal_start = split.vertical_d * n;

    // Check each H_DIM_SIZE block
    size_t num_blocks = split.horizontal_d / skmeans::H_DIM_SIZE;
    for (size_t block = 0; block < num_blocks; ++block) {
        size_t block_start = horizontal_start + block * n * skmeans::H_DIM_SIZE;
        for (size_t vec = 0; vec < n; ++vec) {
            for (size_t dim_in_block = 0; dim_in_block < skmeans::H_DIM_SIZE; ++dim_in_block) {
                size_t orig_dim = split.vertical_d + block * skmeans::H_DIM_SIZE + dim_in_block;
                float expected = static_cast<float>(vec * 10000 + orig_dim);
                size_t pdx_idx = block_start + vec * skmeans::H_DIM_SIZE + dim_in_block;

                EXPECT_FLOAT_EQ(pdx_output[pdx_idx], expected)
                    << "Horizontal block " << block << ", vec " << vec
                    << ", dim_in_block " << dim_in_block << " incorrect";
            }
        }
    }
}

/**
 * @brief Test FULLY_TRANSPOSED mode produces complete column-major output
 */
TEST(PDXLayoutTest, PDXify_FullyTransposed) {
    constexpr size_t TEST_CHUNK_SIZE = 8;
    const size_t d = 128;
    const size_t n = TEST_CHUNK_SIZE;

    // Create identifiable input
    std::vector<float> input(n * d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            input[i * d + j] = static_cast<float>(i * 1000 + j);
        }
    }

    std::vector<float> pdx_output(n * d);
    skmeans::PDXLayout<>::PDXify<true, TEST_CHUNK_SIZE>(
        input.data(), pdx_output.data(), n, d
    );

    // Full transpose: output should be column-major (d x n instead of n x d)
    for (size_t dim = 0; dim < d; ++dim) {
        for (size_t vec = 0; vec < n; ++vec) {
            size_t pdx_idx = dim * n + vec;
            float expected = static_cast<float>(vec * 1000 + dim);
            EXPECT_FLOAT_EQ(pdx_output[pdx_idx], expected)
                << "FULLY_TRANSPOSED failed at dim=" << dim << ", vec=" << vec;
        }
    }
}

} // namespace

