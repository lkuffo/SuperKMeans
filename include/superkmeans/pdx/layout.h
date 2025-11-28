#pragma once

#include "superkmeans/common.h"
#include "superkmeans/pdx/pdx_ivf.h"
#include "superkmeans/pdx/pdxearch.h"
#include <Eigen/Eigen/Dense>
#include <cassert>
#include <memory>
#include <string>

namespace skmeans {

/**
 * @brief Holds the split of dimensions between vertical and horizontal storage.
 */
struct PDXDimensionSplit {
    size_t horizontal_d{0};  ///< Number of horizontal dimensions (stored row-major)
    size_t vertical_d{0};    ///< Number of vertical dimensions (stored column-major)
};

/**
 * @brief PDX data layout manager for efficient SIMD-friendly nearest neighbor search.
 *
 * PDX is a hybrid data layout that splits dimensions into:
 * - Vertical dimensions: stored column-major for efficient early termination scanning
 * - Horizontal dimensions: stored in row-major blocks for efficient SIMD operations
 *
 * This layout enables the PDXearch algorithm to efficiently prune candidates using
 * partial distance computations.
 *
 * @tparam q Quantization type (f32 or u8)
 * @tparam alpha Distance function (l2 or dp)
 */
template <Quantization q = Quantization::f32, DistanceFunction alpha = DistanceFunction::l2>
class PDXLayout {

    using index_t = IndexPDXIVF<q>;
    using scalar_t = skmeans_value_t<q>;
    using cluster_t = Cluster<q>;
    using Pruner = ADSamplingPruner<q>;
    using searcher_t = PDXearch<q, IndexPDXIVF<q>, alpha>;

  public:
    /**
     * @brief Constructs a PDXLayout from existing data buffers.
     *
     * @param pdx_data Pointer to PDX-formatted data buffer
     * @param pruner Reference to the ADSamplingPruner for search operations
     * @param n_points Number of data points
     * @param d Number of dimensions
     * @param hor_data Optional pointer to auxiliary horizontal data for faster pruning
     */
    PDXLayout(scalar_t* pdx_data, Pruner& pruner, size_t n_points, size_t d, scalar_t* hor_data = nullptr) {
        index = std::make_unique<index_t>(); // PDXLayout is owner of the Index
        FromBufferToPDXIndex(pdx_data, n_points, d, hor_data);
        searcher = std::make_unique<searcher_t>(*index, pruner);
    }

    /**
     * @brief Initializes the PDX index structure from a data buffer.
     *
     * Partitions the data into clusters of VECTOR_CHUNK_SIZE vectors each,
     * setting up the index structure for PDXearch operations.
     *
     * @param pdx_data Pointer to PDX-formatted data
     * @param n_points Number of data points
     * @param d Number of dimensions
     * @param hor_data Optional auxiliary horizontal data for the vertical dimensions
     */
    void FromBufferToPDXIndex(
        scalar_t* SKM_RESTRICT pdx_data,
        const size_t n_points,
        const size_t d,
        scalar_t* SKM_RESTRICT hor_data = nullptr
    ) {
        auto [horizontal_d, vertical_d] = GetDimensionSplit(d);
        size_t n_pdx_clusters = n_points / VECTOR_CHUNK_SIZE;
        const size_t full_clusters = n_points / VECTOR_CHUNK_SIZE;
        const size_t n_remaining = n_points % VECTOR_CHUNK_SIZE;
        if (n_remaining) {
            n_pdx_clusters++;
        }

        index->num_clusters = n_pdx_clusters;
        // We define sequential centroid ids
        centroid_ids.resize(n_points);
        std::iota(centroid_ids.begin(), centroid_ids.end(), 0);
        index->num_horizontal_dimensions = horizontal_d;
        index->num_vertical_dimensions = vertical_d;
        index->num_dimensions = d;
        index->is_ivf = false;
        index->is_normalized = false;
        index->clusters.resize(n_pdx_clusters);
        auto pdx_data_p = pdx_data;
        auto hor_data_p = hor_data;
        size_t cluster_idx = 0;
        for (; cluster_idx < full_clusters; cluster_idx++) {
            cluster_t& cluster = index->clusters[cluster_idx];
            cluster.num_embeddings = VECTOR_CHUNK_SIZE;
            cluster.data = pdx_data_p;
            cluster.indices = centroid_ids.data() + (cluster_idx * VECTOR_CHUNK_SIZE);
            if (hor_data_p) {
                cluster.aux_hor_data = hor_data_p;
                hor_data_p += VECTOR_CHUNK_SIZE * vertical_d;
            }
            pdx_data_p += VECTOR_CHUNK_SIZE * d;
        }
        if (n_remaining) {
            cluster_t& cluster = index->clusters[cluster_idx];
            cluster.num_embeddings = n_remaining;
            cluster.data = pdx_data_p;
            cluster.indices = centroid_ids.data() + (cluster_idx * VECTOR_CHUNK_SIZE);
            if (hor_data_p) {
                cluster.aux_hor_data = hor_data_p;
            }
        }
    }

    /**
     * @brief Get number of vertical and horizontal dimensions. We will try to split 25% to
     * vertical and 75% to horizontal.
     *
     * @param d Number of dimensions (cols) in the data
     * @return void
     */
    static inline PDXDimensionSplit GetDimensionSplit(const size_t d) {
        // We compute the split of vertical and horizontal dimensions
        size_t horizontal_d = static_cast<uint32_t>(d * PROPORTION_HORIZONTAL_DIM);
        size_t vertical_d = d - horizontal_d;
        if (horizontal_d % H_DIM_SIZE > 0) {
            horizontal_d = std::floor((1.0 * horizontal_d / H_DIM_SIZE) + 0.5) * H_DIM_SIZE;
            vertical_d = d - horizontal_d;
        }
        if (!vertical_d) {
            horizontal_d = H_DIM_SIZE;
            vertical_d = d - horizontal_d;
        }
        if (d <= H_DIM_SIZE) {
            horizontal_d = 0;
            vertical_d = d;
        }
        return {horizontal_d, vertical_d};
    }

    /**
     * @brief Transform RowMajor matrix to the PDX layout.
     *
     * @tparam q Determines scalar_t
     * @tparam FULLY_TRANSPOSED Whether to do a full ColumMajor transposition every
     * VECTOR_CHUNK_SIZE vectors or the transposition in 25% ColumnMajor and 75% RowMajor dimensions
     * (in blocks of H_DIM_SIZE)
     * @param in_vectors The RowMajor data matrix
     * @param out_pdx_vectors The output buffer of PDX data
     * @param n Number of points (rows) in the data matrix
     * @param d Number of dimensions (cols) in the data matrix
     * @return void
     */
    template <bool FULLY_TRANSPOSED = false, size_t CHUNK_SIZE = VECTOR_CHUNK_SIZE>
    static inline void PDXify(
        const skmeans_value_t<q>* SKM_RESTRICT in_vectors,
        skmeans_value_t<q>* SKM_RESTRICT out_pdx_vectors,
        const size_t n,
        const size_t d
    ) {
        using scalar_t = skmeans_value_t<q>;

        auto [horizontal_d, vertical_d] = GetDimensionSplit(d);
        assert(horizontal_d % H_DIM_SIZE == 0);

        const size_t full_chunks = n / CHUNK_SIZE;
        const size_t n_remaining = n % CHUNK_SIZE;
        // TODO(@lkuffo, high): Parallelize
        for (size_t i = 0; i < full_chunks; ++i) {
            auto chunk_offset = (i * CHUNK_SIZE) * d; // Chunk offset is the same in both layouts
            const scalar_t* SKM_RESTRICT chunk_p = in_vectors + chunk_offset;
            scalar_t* SKM_RESTRICT out_chunk_p = out_pdx_vectors + chunk_offset;
            if constexpr (FULLY_TRANSPOSED) {
                Eigen::Map<
                    const Eigen::Matrix<scalar_t, CHUNK_SIZE, Eigen::Dynamic, Eigen::RowMajor>>
                    in(chunk_p, CHUNK_SIZE, d);
                Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, CHUNK_SIZE, Eigen::RowMajor>>
                    out(out_chunk_p, d, CHUNK_SIZE);
                out.noalias() = in.transpose();
            } else {
                // Vertical Block
                Eigen::Map<
                    const Eigen::Matrix<scalar_t, CHUNK_SIZE, Eigen::Dynamic, Eigen::RowMajor>>
                    in(chunk_p, CHUNK_SIZE, d);
                Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, CHUNK_SIZE, Eigen::RowMajor>>
                    out(out_chunk_p, vertical_d, CHUNK_SIZE);
                out.noalias() = in.leftCols(vertical_d).transpose();
                out_chunk_p += CHUNK_SIZE * vertical_d;

                // Horizontal Blocks
                for (size_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
                    Eigen::Map<Eigen::Matrix<scalar_t, CHUNK_SIZE, H_DIM_SIZE, Eigen::RowMajor>>
                        out_h(out_chunk_p, CHUNK_SIZE, H_DIM_SIZE);
                    out_h.noalias() = in.block(0, vertical_d + j, CHUNK_SIZE, H_DIM_SIZE);
                    out_chunk_p += CHUNK_SIZE * H_DIM_SIZE;
                }
            }
        }
        if (n_remaining) {
            auto chunk_offset = (full_chunks * CHUNK_SIZE) * d; // Chunk offset is the same in both layouts
            const scalar_t* SKM_RESTRICT chunk_p = in_vectors + chunk_offset;
            scalar_t* SKM_RESTRICT out_chunk_p = out_pdx_vectors + chunk_offset;
            if constexpr (FULLY_TRANSPOSED) {
                Eigen::Map<
                    const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    in(chunk_p, n_remaining, d);
                Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    out(out_chunk_p, d, n_remaining);
                out.noalias() = in.transpose();
            } else {
                // Vertical Block
                Eigen::Map<
                    const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    in(chunk_p, n_remaining, d);
                Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                    out(out_chunk_p, vertical_d, n_remaining);
                out.noalias() = in.leftCols(vertical_d).transpose();
                out_chunk_p += n_remaining * vertical_d;

                // Horizontal Blocks
                for (size_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
                    Eigen::Map<Eigen::Matrix<scalar_t, Eigen::Dynamic, H_DIM_SIZE, Eigen::RowMajor>>
                        out_h(out_chunk_p, n_remaining, H_DIM_SIZE);
                    out_h.noalias() = in.block(0, vertical_d + j, n_remaining, H_DIM_SIZE);
                    out_chunk_p += n_remaining * H_DIM_SIZE;
                }
            }
        }
    }

    std::unique_ptr<searcher_t> searcher = nullptr;  ///< PDXearch instance for this layout
    std::unique_ptr<index_t> index;                   ///< Index structure holding cluster metadata

  protected:
    std::vector<uint32_t> centroid_ids;  ///< Vector of centroid IDs (0 to n_points-1)
};

} // namespace skmeans
