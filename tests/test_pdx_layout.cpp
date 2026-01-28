#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/utils.h"

namespace {

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
    size_t actual_chunk_size
) {
    size_t chunk_offset = chunk_idx * CHUNK_SIZE * d;
    if (dim_idx < vertical_d) {
        return pdx_data[chunk_offset + dim_idx * actual_chunk_size + vec_in_chunk];
    } else {
        size_t h_dim = dim_idx - vertical_d;
        size_t block_idx = h_dim / skmeans::H_DIM_SIZE;
        size_t dim_in_block = h_dim % skmeans::H_DIM_SIZE;

        size_t block_start = chunk_offset + vertical_d * actual_chunk_size +
                             block_idx * actual_chunk_size * skmeans::H_DIM_SIZE;
        return pdx_data[block_start + vec_in_chunk * skmeans::H_DIM_SIZE + dim_in_block];
    }
}

/**
 * @brief PDX dimension split correctness for various dimensions
 *
 * Verifies:
 * 1. horizontal_d is a multiple of H_DIM_SIZE (64) or 0
 * 2. horizontal_d + vertical_d = d
 */
TEST(PDXLayoutTest, DimensionSplit_CorrectForVariousDimensions) {
    std::vector<size_t> dimensions = {
        1, 2, 4, 8, 16, 32, 48, 63, 64, 65,
        127, 128, 129, 191, 192, 193, 255, 256, 257,
        384, 512, 600, 768, 900, 1024, 1536, 2048, 3072, 4096
    };

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing d=" + std::to_string(d));

        auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);

        // horizontal_d is a multiple of H_DIM_SIZE or 0
        EXPECT_TRUE(split.horizontal_d % skmeans::H_DIM_SIZE == 0 || split.horizontal_d == 0)
            << "horizontal_d=" << split.horizontal_d
            << " is not a multiple of " << skmeans::H_DIM_SIZE << " (or 0) for d=" << d;

        // horizontal_d + vertical_d = d
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

/**
 * @brief Test PDXify preserves all data values (no data loss)
 *
 * After PDXify, we should be able to retrieve every original value
 * by accessing the correct position in the PDX layout.
 */
TEST(PDXLayoutTest, PDXify_PreservesAllValues) {
    constexpr size_t TEST_CHUNK_SIZE = 64;

    std::vector<size_t> dimensions = {128, 256, 384, 512};
    const size_t n = TEST_CHUNK_SIZE * 2 + 17;  // 2 full chunks + partial

    for (size_t d : dimensions) {
        SCOPED_TRACE("Testing PDXify d=" + std::to_string(d) + ", n=" + std::to_string(n));

        auto split = skmeans::PDXLayout<>::GetDimensionSplit(d);
        if (split.horizontal_d == 0) continue;
        auto input = skmeans::GenerateRandomVectors(n, d, -10.0f, 10.0f, 42);
        std::vector<float> pdx_output(n * d);

        skmeans::PDXLayout<>::PDXify<false, TEST_CHUNK_SIZE>(
            input.data(), pdx_output.data(), n, d
        );

        size_t full_chunks = n / TEST_CHUNK_SIZE;
        size_t remaining = n % TEST_CHUNK_SIZE;
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
 * @brief Test FULLY_TRANSPOSED mode produces complete column-major output
 */
TEST(PDXLayoutTest, PDXify_FullyTransposed) {
    constexpr size_t TEST_CHUNK_SIZE = 8;
    const size_t d = 128;
    const size_t n = TEST_CHUNK_SIZE;

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

