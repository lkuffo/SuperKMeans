#ifndef SKMEANS_PDX_LAYOUT_HPP
#define SKMEANS_PDX_LAYOUT_HPP

#include "superkmeans/common.h"
#include "superkmeans/pdx/index_base/pdx_ivf.hpp"
#include "superkmeans/pdx/pdxearch.h"
#include <Eigen/Eigen/Dense>
#include <cassert>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace skmeans {

struct PDXDimensionSplit {
    size_t horizontal_d{0};
    size_t vertical_d{0};
};

template <Quantization q = f32, DistanceFunction alpha = l2, class Pruner = ADSamplingPruner<q>>
class PDXLayout {

    using index_t = IndexPDXIVF<q>;
    using scalar_t = skmeans_value_t<q>;
    using cluster_t = Cluster<q>;
    using searcher_t = PDXearch<q, IndexPDXIVF<q>, alpha, Pruner>;

  public:
    PDXLayout(scalar_t* pdx_data, Pruner& pruner, size_t n_points, size_t d) {
        index = std::make_unique<index_t>(); // PDXLayout is owner of the Index
        FromBufferToPDXIndex(pdx_data, n_points, d);
        searcher = std::make_unique<searcher_t>(*index, pruner);
    }

    PDXLayout(scalar_t* pdx_data, Pruner& pruner, size_t n_points, size_t d, size_t partial_d, scalar_t* hor_data) {
        index = std::make_unique<index_t>(); // PDXLayout is owner of the Index
        FromBufferToPDXIndex(pdx_data, n_points, d, partial_d, hor_data);
        searcher = std::make_unique<searcher_t>(*index, pruner);
    }


    // TODO(@lkuffo, low): Support arbitrary cluster sizes rather than always 64
    void FromBufferToPDXIndex(
        scalar_t* SKM_RESTRICT pdx_data,
        const size_t n_points,
        const size_t d
    ) {
        // TODO(@lkuffo, high): Support cluster sizes that are not multiples of 64
        assert(n_points % VECTOR_CHUNK_SIZE == 0);

        auto [horizontal_d, vertical_d] = GetDimensionSplit(d);
        size_t n_pdx_clusters = n_points / VECTOR_CHUNK_SIZE;
        index->num_clusters = n_pdx_clusters;
        // TODO(@lkuffo, high): Does this belong here?
        // Seems to important to define the centroid ids here
        centroid_ids.resize(n_points);
        std::iota(centroid_ids.begin(), centroid_ids.end(), 0);
        index->num_horizontal_dimensions = horizontal_d;
        index->num_vertical_dimensions = vertical_d;
        index->num_dimensions = d;
        index->is_ivf = false;
        index->is_normalized = false;
        index->clusters.resize(n_pdx_clusters);
        auto pdx_data_p = pdx_data;
        size_t cluster_idx = 0;
        for (size_t cluster_offset = 0; cluster_offset < n_points;
             cluster_offset += VECTOR_CHUNK_SIZE) {
            cluster_t& cluster = index->clusters[cluster_idx];
            cluster.num_embeddings = VECTOR_CHUNK_SIZE;
            cluster.data = pdx_data_p;
            cluster.indices = centroid_ids.data() + cluster_offset;
            pdx_data_p += VECTOR_CHUNK_SIZE * d;
            cluster_idx += 1;
        }
    }

    void FromBufferToPDXIndex(
        scalar_t* SKM_RESTRICT pdx_data,
        const size_t n_points,
        const size_t d,
        const size_t partial_d,
        scalar_t* SKM_RESTRICT hor_data
    ) {
        // TODO(@lkuffo, high): Support cluster sizes that are not multiples of 64
        assert(n_points % VECTOR_CHUNK_SIZE == 0);

        auto [horizontal_d, vertical_d] = GetDimensionSplit(d);
        assert(vertical_d >= partial_d);
        size_t n_pdx_clusters = n_points / VECTOR_CHUNK_SIZE;
        index->num_clusters = n_pdx_clusters;
        // TODO(@lkuffo, high): Does this belong here?
        // Seems to important to define the centroid ids here
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
        for (size_t cluster_offset = 0; cluster_offset < n_points;
             cluster_offset += VECTOR_CHUNK_SIZE) {
            cluster_t& cluster = index->clusters[cluster_idx];
            cluster.num_embeddings = VECTOR_CHUNK_SIZE;
            cluster.data = pdx_data_p;
            cluster.indices = centroid_ids.data() + cluster_offset;
            cluster.aux_hor_data = hor_data_p;
            pdx_data_p += VECTOR_CHUNK_SIZE * d;
            hor_data_p +=
                VECTOR_CHUNK_SIZE * (vertical_d - partial_d
                                    ); // Contains only the vertical dimensions not visited by BLAS
            cluster_idx += 1;
        }
    }

    /**
     * @brief Get number of vertical and horizontal dimensions. We will try to split 25% to
     * vertical and 75% to horizontal. This function will always achieve horizontal blocks
     * of 64 values
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
        // TODO(@lkuffo): What are the consequences of this?
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
        assert(n % CHUNK_SIZE == 0);

        auto [horizontal_d, vertical_d] = GetDimensionSplit(d);

        // TODO(@lkuffo, high): Parallelize
        for (size_t i = 0; i < n; i += CHUNK_SIZE) {
            auto chunk_offset = i * d; // Chunk offset is the same in both layouts
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
                    Eigen::Map<Eigen::Matrix<scalar_t, H_DIM_SIZE, CHUNK_SIZE, Eigen::RowMajor>>
                        out_h(out_chunk_p, H_DIM_SIZE, CHUNK_SIZE);
                    out_h.noalias() = in.block(0, vertical_d + j, CHUNK_SIZE, H_DIM_SIZE);
                    out_chunk_p += CHUNK_SIZE * H_DIM_SIZE;
                }
            }
        }
    }

    template <bool FULLY_TRANSPOSED = false>
    static inline void UpdateSum(
        scalar_t* SKM_RESTRICT pdx_data,
        const scalar_t* SKM_RESTRICT vector,
        const size_t position,
        const size_t n_points,
        const size_t d,
        const size_t horizontal_d,
        const size_t vertical_d
    ) {
        // Calculate Position
        // Apply operand
        //
    }

    SKM_ALWAYS_INLINE static size_t OffsetCalculator(
        const size_t vector_position,
        const size_t d,
        const size_t horizontal_d,
        const size_t vertical_d
    ) {}

    template <typename T>
    static bool CheckBlockTranspose(const T* in_vectors, const T* out_vectors, size_t n, size_t d) {
        constexpr float EPS = 1e-6f;
        assert(n % VECTOR_CHUNK_SIZE == 0);

        for (size_t i = 0; i < n; i += VECTOR_CHUNK_SIZE) {
            for (size_t row = 0; row < VECTOR_CHUNK_SIZE; ++row) {
                for (size_t col = 0; col < d; ++col) {
                    // Original linear index in row-major input
                    size_t in_idx = (i + row) * d + col;
                    // In the output, the chunk is transposed: out shape is (d x VECTOR_CHUNK_SIZE)
                    size_t out_idx = i * d + col * VECTOR_CHUNK_SIZE + row;
                    if (std::fabs(in_vectors[in_idx] - out_vectors[out_idx]) > EPS) {
                        std::cerr << "Mismatch at input(" << (i + row) << "," << col
                                  << ") = " << in_vectors[in_idx]
                                  << " vs output = " << out_vectors[out_idx] << "\n";
                        throw std::runtime_error("CheckBlockTranspose: Incorrect input");
                    }
                }
            }
        }
        return true;
    }

    template <typename T>
    static bool CheckBlockTransposeNonFull(
        const T* in_vectors,
        const T* out_vectors,
        size_t n,
        size_t d,
        size_t horizontal_d,
        size_t vertical_d
    ) {
        static_assert(std::is_arithmetic<T>::value, "T must be arithmetic");
        constexpr double EPS_DBL = 1e-6;
        constexpr float EPS_FLT = 1e-6f;
        const double EPS = std::is_same<T, double>::value ? EPS_DBL : EPS_FLT;

        assert(n % VECTOR_CHUNK_SIZE == 0 && "n must be multiple of VECTOR_CHUNK_SIZE");

        const size_t num_horizontal_blocks = (horizontal_d / H_DIM_SIZE);

        for (size_t i = 0; i < n; i += VECTOR_CHUNK_SIZE) {
            // chunk base addresses
            const size_t chunk_offset = i * d;
            const T* chunk_in =
                in_vectors + chunk_offset; // element (row = i + r, col = c) -> chunk_in[r*d + c]
            const T* chunk_out =
                out_vectors + chunk_offset; // out layout described above, flattened.

            // 1) Vertical part: out_rows = vertical_d, out_cols = VECTOR_CHUNK_SIZE
            for (size_t out_col = 0; out_col < vertical_d; ++out_col) {
                for (size_t out_row = 0; out_row < VECTOR_CHUNK_SIZE; ++out_row) {
                    const size_t in_row = out_row;
                    const size_t in_col = out_col;
                    const T a = chunk_in[in_row * d + in_col];
                    const T b = chunk_out[out_col * VECTOR_CHUNK_SIZE + out_row];
                    if (std::fabs(static_cast<double>(a) - static_cast<double>(b)) > EPS) {
                        std::cerr << "PDX check mismatch (vertical) chunk_start_row=" << i
                                  << " input(" << (i + in_row) << "," << in_col << ") = " << a
                                  << " vs out[col=" << out_col << ", row=" << out_row << "] = " << b
                                  << "\n";
                        throw std::runtime_error(
                            "CheckPDXifyV2_NonFullyTransposed: mismatch vertical"
                        );
                    }
                }
            }

            // 2) Horizontal blocks. blocks start at input column base = vertical_d
            for (size_t blk = 0; blk < num_horizontal_blocks; ++blk) {
                const size_t in_block_col0 = vertical_d + (blk * H_DIM_SIZE);
                constexpr size_t block_width = H_DIM_SIZE;

                // For each element in the transposed block
                for (size_t br = 0; br < block_width; ++br) {
                    for (size_t bc = 0; bc < VECTOR_CHUNK_SIZE; ++bc) {
                        const size_t in_row = i + bc;
                        const size_t in_col = in_block_col0 + br;
                        const size_t in_idx = in_row * d + in_col;

                        // compute out_idx:
                        // output block starts at chunk_offset + out_block_start * VECTOR_CHUNK_SIZE
                        // out_block_start (in columns) is blk*H_DIM_SIZE + vertical_d, but we
                        // already advanced vertical_d earlier
                        const size_t out_row = br; // 0..block_width-1
                        const size_t out_col =
                            (blk * H_DIM_SIZE) + bc; // position within horizontal region
                        const size_t out_idx =
                            chunk_offset + (vertical_d + out_col) * VECTOR_CHUNK_SIZE + out_row;

                        const double a = static_cast<double>(in_vectors[in_idx]);
                        const double b = static_cast<double>(out_vectors[out_idx]);
                        if (std::fabs(a - b) > EPS) {
                            std::cerr << "PDX check mismatch (horizontal block) at chunk " << i
                                      << " input(" << in_row << "," << in_col << ")[" << in_idx
                                      << "] = " << a << " vs out[" << out_idx << "] = " << b
                                      << "\n";
                            throw std::runtime_error(
                                "CheckBlockTransposeNonFull: mismatch horizontal block"
                            );
                        }
                    }
                }
            }
        } // end chunk loop
        return true; // all good
    }

    std::unique_ptr<searcher_t> searcher = nullptr;
    std::unique_ptr<index_t> index;

  protected:
    std::vector<uint32_t> centroid_ids;
};

} // namespace skmeans

#endif // SKMEANS_PDX_LAYOUT_HPP
