#pragma once

#include "superkmeans/common.h"
#include "superkmeans/profiler.h"
#include "superkmeans/quantizers/quantizer.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstring>
#include <limits>
#include <memory>
#include <omp.h>
#include <vector>

#include <faiss/impl/ProductQuantizer.h>

namespace skmeans {

/**
 * @brief 8-bit Product Quantizer (Ks=256) with SDC distance and sparse voting.
 *
 * Splits d-dimensional vectors into M subspaces of dsub=d/M dimensions.
 * Each subspace is encoded as a uint8 index into a 256-entry codebook trained
 * by FAISS ProductQuantizer. Distance computation uses Symmetric Distance
 * Computation (SDC): precomputed pairwise codeword distances per subspace.
 *
 * SDC distance is exact L2² between reconstructed vectors:
 *   SDC(a,b) = Σ_m ||cw_m[a_m] - cw_m[b_m]||² = ||decode(a) - decode(b)||²
 *
 * Centroid updates use sparse voting (Section 4.2 of PQ-means paper):
 * per subspace, build frequency histograms and pick the codeword minimizing
 * weighted SDC distance to all cluster members, accelerated via BLAS sgemm.
 */
class PQ8Quantizer : public IQuantizer<Quantization::u8> {
  public:
    using quantized_t = IQuantizer::quantized_t;

    static constexpr size_t Ks = 256;

    explicit PQ8Quantizer(size_t M) : M_(M) {}

    void Fit(const float* data, size_t n, size_t d) override {
        SKM_PROFILE_SCOPE("fitting");
        assert(d % M_ == 0 && "d must be divisible by M");
        d_ = d;
        dsub_ = d / M_;

        faiss_pq_ = std::make_unique<faiss::ProductQuantizer>(d, M_, 8);
        faiss_pq_->verbose = true;
        faiss_pq_->train(n, data);
        faiss_pq_->compute_sdc_table();

        // Copy SDC table: layout [M × Ks × Ks]
        sdc_table_ = faiss_pq_->sdc_table;
        assert(sdc_table_.size() == M_ * Ks * Ks);

        fitted = true;
    }

    void Encode(const float* in, quantized_t* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("encoding");
        assert(fitted);
        assert(d == d_);
        faiss_pq_->compute_codes(in, out, n);
    }

    void Decode(const quantized_t* in, float* out, size_t n, size_t d) const override {
        SKM_PROFILE_SCOPE("decoding");
        assert(fitted);
        assert(d == d_);
        faiss_pq_->decode(in, out, n);
    }

    size_t CodeSize(size_t /*d*/) const override { return M_; }

    /**
     * @brief Norms are not used for SDC distance — fill with zeros.
     *
     * SDC computes exact L2² directly from codes without the norm expansion trick.
     */
    void ComputeNorms(
        const quantized_t* /*data*/, size_t n, size_t /*d*/, float* out_norms
    ) const override {
        std::fill_n(out_norms, n, 0.0f);
    }

    /**
     * @brief Find top-1 nearest neighbor using exact SDC with contiguous flat LUT.
     *
     * Per data point, builds a contiguous M × 256 float LUT in L1 cache,
     * then scans centroids with 4-way unrolling (FAISS gather pattern).
     * Distances are exact (no quantization or approximation).
     */
    void FindNearestNeighbor(
        const quantized_t* x,
        const quantized_t* y,
        const float* /*x_float*/,
        const float* /*y_float*/,
        size_t n_x,
        size_t n_y,
        size_t /*d*/,
        const float* /*norms_x*/,
        const float* /*norms_y*/,
        uint32_t* out_knn,
        float* out_distances,
        float* /*tmp_buf*/
    ) const override {
        SKM_PROFILE_SCOPE("search");
        assert(fitted);

#pragma omp parallel num_threads(g_n_threads)
        {
            // Per-thread contiguous LUT: M × 256 floats (fits in L1 for typical M)
            std::vector<float> flat_lut(M_ * Ks);

#pragma omp for schedule(dynamic, 256)
            for (size_t i = 0; i < n_x; ++i) {
                const quantized_t* xi = x + i * M_;

                // Build contiguous flat LUT from scattered SDC rows
                for (size_t m = 0; m < M_; ++m) {
                    std::memcpy(
                        flat_lut.data() + m * Ks,
                        sdc_table_.data() + m * Ks * Ks + xi[m] * Ks,
                        Ks * sizeof(float)
                    );
                }

                float best_dist = std::numeric_limits<float>::max();
                uint32_t best_j = 0;

                // 4-way unrolled centroid scan
                const size_t n_y4 = n_y - (n_y % 4);
                for (size_t j = 0; j < n_y4; j += 4) {
                    const quantized_t* y0 = y + (j + 0) * M_;
                    const quantized_t* y1 = y + (j + 1) * M_;
                    const quantized_t* y2 = y + (j + 2) * M_;
                    const quantized_t* y3 = y + (j + 3) * M_;
                    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                    for (size_t m = 0; m < M_; ++m) {
                        const float* lut_m = flat_lut.data() + m * Ks;
                        d0 += lut_m[y0[m]];
                        d1 += lut_m[y1[m]];
                        d2 += lut_m[y2[m]];
                        d3 += lut_m[y3[m]];
                    }
                    if (d0 < best_dist) { best_dist = d0; best_j = static_cast<uint32_t>(j + 0); }
                    if (d1 < best_dist) { best_dist = d1; best_j = static_cast<uint32_t>(j + 1); }
                    if (d2 < best_dist) { best_dist = d2; best_j = static_cast<uint32_t>(j + 2); }
                    if (d3 < best_dist) { best_dist = d3; best_j = static_cast<uint32_t>(j + 3); }
                }
                // Remainder
                for (size_t j = n_y4; j < n_y; ++j) {
                    const quantized_t* yj = y + j * M_;
                    float dist = 0.0f;
                    for (size_t m = 0; m < M_; ++m) {
                        dist += flat_lut[m * Ks + yj[m]];
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_j = static_cast<uint32_t>(j);
                    }
                }

                out_knn[i] = best_j;
                out_distances[i] = best_dist;
            }
        }
    }

    /**
     * @brief Update centroids via sparse voting in the PQ code domain.
     *
     * For each cluster and each subspace m:
     * 1. Build a frequency histogram freq[Ks] counting how many cluster members
     *    have each codeword in subspace m.
     * 2. Compute weighted votes: votes[k] = Σ_k' freq[k'] * sdc[m][k'][k]
     *    This is the total SDC distance from codeword k to all cluster members.
     * 3. Pick argmin(votes) as the new centroid code for subspace m.
     *
     * Step 2 is batched across all clusters as a BLAS sgemm:
     *   votes[n_clusters × Ks] = freq[n_clusters × Ks] × SDC_m[Ks × Ks]
     */
    void SparseVotingUpdate(
        const quantized_t* codes, const uint32_t* assignments,
        quantized_t* out_centroids, uint32_t* out_cluster_sizes,
        size_t n, size_t n_clusters, size_t /*d*/
    ) const override {
        SKM_PROFILE_SCOPE("sparse_voting");
        assert(fitted);

        // Count cluster sizes and build per-cluster member lists
        std::fill_n(out_cluster_sizes, n_clusters, 0u);
        for (size_t i = 0; i < n; ++i) {
            out_cluster_sizes[assignments[i]]++;
        }

        // Frequency matrix: freq[cluster × Ks] per subspace
        // We process one subspace at a time to keep memory bounded
        std::vector<float> freq(n_clusters * Ks);
        std::vector<float> votes(n_clusters * Ks);

        for (size_t m = 0; m < M_; ++m) {
            // Build frequency histograms for subspace m
            std::fill(freq.begin(), freq.end(), 0.0f);
            for (size_t i = 0; i < n; ++i) {
                uint32_t cluster = assignments[i];
                uint8_t code = codes[i * M_ + m];
                freq[cluster * Ks + code] += 1.0f;
            }

            // BLAS sgemm: votes = freq × SDC_m
            // freq:  [n_clusters × Ks]  (row-major)
            // SDC_m: [Ks × Ks]          (row-major)
            // votes: [n_clusters × Ks]  (row-major)
            //
            // Fortran column-major: C = alpha * A * B + beta * C
            // With row-major matrices: C^T = alpha * B^T * A^T + beta * C^T
            // So we pass: transa='N', transb='N', m=Ks, n=n_clusters, k=Ks
            int blas_m = static_cast<int>(Ks);
            int blas_n = static_cast<int>(n_clusters);
            int blas_k = static_cast<int>(Ks);
            float alpha = 1.0f;
            float beta = 0.0f;
            int lda = static_cast<int>(Ks);
            int ldb = static_cast<int>(Ks);
            int ldc = static_cast<int>(Ks);

            sgemm_(
                "N", "N",
                &blas_m, &blas_n, &blas_k,
                &alpha,
                sdc_table_.data() + m * Ks * Ks, &lda,
                freq.data(), &ldb,
                &beta,
                votes.data(), &ldc
            );

            // Pick argmin per cluster
#pragma omp parallel for num_threads(g_n_threads)
            for (size_t c = 0; c < n_clusters; ++c) {
                if (out_cluster_sizes[c] == 0) {
                    out_centroids[c * M_ + m] = 0;
                    continue;
                }
                const float* row = votes.data() + c * Ks;
                size_t best_k = 0;
                float best_vote = row[0];
                for (size_t k = 1; k < Ks; ++k) {
                    if (row[k] < best_vote) {
                        best_vote = row[k];
                        best_k = k;
                    }
                }
                out_centroids[c * M_ + m] = static_cast<uint8_t>(best_k);
            }
        }
    }

    void AverageCentroids(
        const uint32_t* /*accumulators*/,
        const uint32_t* /*cluster_sizes*/,
        quantized_t* /*out*/,
        size_t /*n_clusters*/,
        size_t /*d*/
    ) const override {
        // Not used — PQ uses SparseVotingUpdate instead
    }

    bool IsFitted() const override { return fitted; }
    bool SupportsPruning() const override { return false; }
    bool UsesSparseVoting() const override { return true; }

  private:
    size_t M_;
    size_t dsub_ = 0;
    size_t d_ = 0;
    bool fitted = false;
    std::unique_ptr<faiss::ProductQuantizer> faiss_pq_;
    std::vector<float> sdc_table_; // [M_ × Ks × Ks]
};

} // namespace skmeans
