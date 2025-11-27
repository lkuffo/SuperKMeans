#ifndef SKMEANS_ADSAMPLING_HPP
#define SKMEANS_ADSAMPLING_HPP

#include "superkmeans/distance_computers/base_computers.hpp"
#include "superkmeans/pdx/utils.h"

#include <Eigen/Eigen/Dense>
#include <queue>
#ifdef HAS_FFTW
#include <fftw3.h>
#endif

namespace skmeans {

/******************************************************************
 * ADSampling pruner
 ******************************************************************/
template <Quantization q = f32>
class ADSamplingPruner {
    using DISTANCES_TYPE = skmeans_distance_t<q>;
    using value_t = skmeans_value_t<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  public:
    uint32_t num_dimensions;
    std::vector<float> ratios{};

    ADSamplingPruner(uint32_t num_dimensions_, float epsilon0)
        : num_dimensions(num_dimensions_), epsilon0(epsilon0) {
        InitializeRatios();
        bool matrix_created = false;
#ifdef HAS_FFTW
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
            matrix.resize(1, num_dimensions);
            std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<int> dist(0, 1);
            for (size_t i = 0; i < num_dimensions; ++i) {
                matrix(i) = dist(gen) ? 1.0f : -1.0f;
            }
            flip_masks.resize(num_dimensions);
            for (size_t i = 0; i < num_dimensions; ++i) {
                flip_masks[i] = (flip_masks[i] < 0.0 ? 0x80000000u : 0u);
            }
            matrix_created = true;
        }
#endif
        if (!matrix_created) {
            matrix.resize(num_dimensions, num_dimensions);
            matrix = MatrixR::NullaryExpr(num_dimensions, num_dimensions, []() {
                static thread_local std::mt19937 gen(std::random_device{}());
                static thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
                return dist(gen);
            });
            const Eigen::HouseholderQR<MatrixR> qr(matrix);
            matrix = qr.householderQ() * MatrixR::Identity(num_dimensions, num_dimensions);
        }
    }

    ADSamplingPruner(uint32_t num_dimensions, float epsilon0, float* matrix_p)
        : num_dimensions(num_dimensions), epsilon0(epsilon0) {
        InitializeRatios();
#ifdef HAS_FFTW
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
            matrix =
                Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                    matrix_p, 1, num_dimensions
                );
        } else {
            matrix =
                Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                    matrix_p, num_dimensions, num_dimensions
                );
        }
#else
        matrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            matrix_p, num_dimensions, num_dimensions
        );
#endif
    }

    void InitializeRatios() {
        // + 1 to be able to map n_dims to 1.0f, avoiding a branch in GetPruningThreshold
        ratios.resize(num_dimensions + 1);
        for (size_t i = 0; i < num_dimensions + 1; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    static bool VerifyOrthonormal(const MatrixR& Q, float tol = 1e-5f) {
        uint32_t n = Q.rows();
        assert(Q.rows() == Q.cols());
        // Compute Q * Q^T (use column-major temporary for stable multiply)
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Qc = Q; // conversion
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> M = Qc * Qc.transpose();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> I =
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
        float err = (M - I).norm(); // Frobenius norm
        return err <= tol;
    }

    void SetEpsilon0(float epsilon0) {
        ADSamplingPruner::epsilon0 = epsilon0;
        for (size_t i = 0; i < num_dimensions + 1; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    void SetMatrix(const Eigen::MatrixXf& matrix) { ADSamplingPruner::matrix = matrix; }

    template <Quantization Q = q>
    SKM_NO_INLINE
    skmeans_distance_t<Q> GetPruningThreshold(
        const KNNCandidate<Q>& best_candidate,
        const uint32_t current_dimension_idx
    ) {
        return best_candidate.distance * ratios[current_dimension_idx];
    }

    void PreprocessQuery(const float* raw_query, float* query) {
        Multiply(raw_query, query, num_dimensions);
    }

    void FlipSign(const float* data, float* out, const size_t n) {
#pragma omp parallel for num_threads(g_n_threads)
        for (size_t i = 0; i < n; ++i) {
            const size_t offset = i * num_dimensions;
            UtilsComputer<q>::FlipSign(data + offset, out + offset, flip_masks.data(), num_dimensions);
        }
    }

    void Rotate(
        const value_t* SKM_RESTRICT vectors,
        value_t* SKM_RESTRICT out_buffer,
        const uint32_t n
    ) {
        TicToc m;
        m.Tic();
        Eigen::Map<const MatrixR> vectors_matrix(vectors, n, num_dimensions);
        Eigen::Map<MatrixR> out(out_buffer, n, num_dimensions);
#ifdef HAS_FFTW
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
            FlipSign(vectors, out_buffer, n);
            fftwf_init_threads();
            fftwf_plan_with_nthreads(g_n_threads);
            int n0 = static_cast<int>(num_dimensions); // length of each 1D transform
            int howmany = static_cast<int>(n); // number of transforms (one per row)
            fftw_r2r_kind kind[1] = {FFTW_REDFT10};
            auto flag = FFTW_MEASURE;
            if (IsPowerOf2(num_dimensions)) { 
                flag = FFTW_ESTIMATE;
            }
            fftwf_plan plan = fftwf_plan_many_r2r(
                1,
                &n0,
                howmany,
                out.data(), /*in*/
                NULL,
                1,
                n0,         /*inembed, istride, idist*/
                out.data(), /*out*/
                NULL,
                1,
                n0, /*onembed, ostride, odist*/
                kind,
                flag
            );
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
            const float s0 = std::sqrt(1.0f / (4.0f * num_dimensions));
            const float s = std::sqrt(1.0f / (2.0f * num_dimensions));
            out.col(0) *= s0;
            out.rightCols(num_dimensions - 1) *= s;
            return;
        }
#endif
        out.noalias() = vectors_matrix * matrix.transpose();
    }

    /**
     * @brief Unrotate vectors (inverse of Rotate).
     * For orthonormal matrix Q: Rotate does out = vectors * Q^T, so Unrotate does out = vectors * Q
     * For DCT: applies inverse scaling, then inverse DCT (DCT-III), then FlipSign
     */
    void Unrotate(
        const float* SKM_RESTRICT rotated_vectors,
        float* SKM_RESTRICT out_buffer,
        const uint32_t n
    ) {
        Eigen::Map<const MatrixR> vectors_matrix(rotated_vectors, n, num_dimensions);
        Eigen::Map<MatrixR> out(out_buffer, n, num_dimensions);
#ifdef HAS_FFTW
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
            // Copy input to output buffer for in-place transform
            std::memcpy(out_buffer, rotated_vectors, n * num_dimensions * sizeof(float));
            
            // Undo scaling (inverse of forward scaling)
            const float inv_s0 = std::sqrt(4.0f * num_dimensions);
            const float inv_s = std::sqrt(2.0f * num_dimensions);
            out.col(0) *= inv_s0;
            out.rightCols(num_dimensions - 1) *= inv_s;
            
            // Apply inverse DCT (DCT-III = FFTW_REDFT01)
            fftwf_init_threads();
            fftwf_plan_with_nthreads(g_n_threads);
            int n0 = static_cast<int>(num_dimensions);
            int howmany = static_cast<int>(n);
            fftw_r2r_kind kind[1] = {FFTW_REDFT01};  // DCT-III (inverse of DCT-II)
            auto flag = FFTW_MEASURE;
            if (IsPowerOf2(num_dimensions)) {
                flag = FFTW_ESTIMATE;
            }
            fftwf_plan plan = fftwf_plan_many_r2r(
                1,
                &n0,
                howmany,
                out.data(),
                NULL, 1, n0,
                out.data(),
                NULL, 1, n0,
                kind,
                flag
            );
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
            
            // FFTW's DCT-III needs normalization by 1/(2*n)
            out *= (1.0f / (2.0f * num_dimensions));
            
            // Undo FlipSign (FlipSign is its own inverse)
            FlipSign(out_buffer, out_buffer, n);
            return;
        }
#endif
        // For orthonormal matrix: Q^{-1} = Q^T, and Rotate does v * Q^T, so Unrotate does v * Q
        out.noalias() = vectors_matrix * matrix;
    }

  private:
    float epsilon0 = 2.1;
    MatrixR matrix;
    std::vector<uint32_t> flip_masks;

    float GetRatio(const size_t& visited_dimensions) {
        if (visited_dimensions == 0) {
            return 1;
        }
        if (visited_dimensions == (int) num_dimensions) {
            return 1.0;
        }
        return 1.0 * visited_dimensions / ((int) num_dimensions) *
               (1.0 + epsilon0 / std::sqrt(visited_dimensions)) *
               (1.0 + epsilon0 / std::sqrt(visited_dimensions));
    }

    void Multiply(const float* raw_query, float* query, uint32_t num_dimensions) {
        Eigen::Map<const Eigen::RowVectorXf> query_matrix(raw_query, num_dimensions);
        Eigen::Map<Eigen::RowVectorXf> output(query, num_dimensions);
#ifdef HAS_FFTW
        if (num_dimensions >= D_THRESHOLD_FOR_DCT_ROTATION) {
            Eigen::RowVectorXf first_row = matrix.row(0);
            Eigen::RowVectorXf pre_output = query_matrix.array() * first_row.array();
            fftwf_plan plan = fftwf_plan_r2r_1d(
                num_dimensions, pre_output.data(), output.data(), FFTW_REDFT10, FFTW_ESTIMATE
            );
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
            output[0] *= std::sqrt(1.0 / (4 * num_dimensions));
            for (int i = 1; i < num_dimensions; ++i)
                output[i] *= std::sqrt(1.0 / (2 * num_dimensions));
            return;
        }
#endif
        output.noalias() = query_matrix * matrix;
    }
};

} // namespace skmeans

#endif // SKMEANS_ADSAMPLING_HPP
