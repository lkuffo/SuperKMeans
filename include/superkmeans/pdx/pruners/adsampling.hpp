#ifndef SKMEANS_ADSAMPLING_HPP
#define SKMEANS_ADSAMPLING_HPP

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
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;
    // TODO(@lkuffo): Rename
    using MatrixF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  public:
    uint32_t num_dimensions;
    std::vector<float> ratios{};

    ADSamplingPruner(uint32_t num_dimensions_, float epsilon0)
        : num_dimensions(num_dimensions_), epsilon0(epsilon0) {
        InitializeRatios();
#ifdef HAS_FFTW
        // TODO(@lkuffo) Implement FFTW matrix
#else
        matrix.resize(num_dimensions, num_dimensions);
        matrix = MatrixF::NullaryExpr(num_dimensions, num_dimensions, []() {
            static thread_local std::mt19937 gen(std::random_device{}());
            static thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
            return dist(gen);
        });
        const Eigen::HouseholderQR<MatrixF> qr(matrix);
        matrix = qr.householderQ() * MatrixF::Identity(num_dimensions, num_dimensions);
#endif
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
        ratios.resize(num_dimensions);
        for (size_t i = 0; i < num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    static bool VerifyOrthonormal(const MatrixF& Q, float tol = 1e-5f) {
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
        for (size_t i = 0; i < num_dimensions; ++i) {
            ratios[i] = GetRatio(i);
        }
    }

    void SetMatrix(const Eigen::MatrixXf& matrix) { ADSamplingPruner::matrix = matrix; }

    template <Quantization Q = q>
    skmeans_distance_t<Q> GetPruningThreshold(
        uint32_t k,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>>&
            heap,
        const uint32_t current_dimension_idx
    ) {
        float ratio = current_dimension_idx == num_dimensions ? 1 : ratios[current_dimension_idx];
        return heap.top().distance * ratio;
    }

    void PreprocessQuery(float* raw_query, float* query) {
        Multiply(raw_query, query, num_dimensions);
    }

    // TODO(@lkuffo, high): Pararellize
    void Rotate( float* SKM_RESTRICT vectors, float* SKM_RESTRICT out_buffer, uint32_t n) {
        Eigen::Map<const MatrixF> vectors_matrix(vectors, n, num_dimensions);
        Eigen::Map<MatrixF> out(out_buffer, n, num_dimensions);
        out.noalias() = vectors_matrix * matrix.transpose();
    }

  private:
    float epsilon0 = 2.1;
    MatrixF matrix;

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

    void Multiply(float* raw_query, float* query, uint32_t num_dimensions) {
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
