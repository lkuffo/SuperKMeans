#undef HAS_FFTW

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <omp.h>
#include <tuple>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

namespace {

// Test data stored in: tests/test_data.bin
// Regenerate with: ./generate_wcss_ground_truth.out

// clang-format off
const std::map<std::pair<size_t, size_t>, float> GROUND_TRUTH = {
    // k=10
    {{10, 4}, 1.31454e+06f},
    {{10, 16}, 1.04449e+07f},
    {{10, 32}, 2.47275e+07f},
    {{10, 64}, 5.36638e+07f},
    {{10, 100}, 8.73695e+07f},
    {{10, 128}, 1.12716e+08f},
    {{10, 384}, 3.46747e+08f},
    {{10, 512}, 4.64129e+08f},
    {{10, 600}, 5.45487e+08f},
    {{10, 768}, 6.97502e+08f},
    // k=100
    {{100, 4}, 8.52101e+04f},
    {{100, 16}, 1.52457e+06f},
    {{100, 32}, 4.87128e+06f},
    {{100, 64}, 9.18784e+06f},
    {{100, 100}, 1.80314e+07f},
    {{100, 128}, 2.60595e+07f},
    {{100, 384}, 8.83893e+07f},
    {{100, 512}, 1.17779e+08f},
    {{100, 600}, 1.35796e+08f},
    {{100, 768}, 1.92280e+08f},
    // k=250
    {{250, 4}, 3.68675e+04f},
    {{250, 16}, 2.15290e+05f},
    {{250, 32}, 2.99748e+05f},
    {{250, 64}, 6.09428e+05f},
    {{250, 100}, 9.59432e+05f},
    {{250, 128}, 2.00011e+06f},
    {{250, 384}, 6.95437e+06f},
    {{250, 512}, 4.97565e+06f},
    {{250, 600}, 5.83501e+06f},
    {{250, 768}, 7.47213e+06f}
};
// clang-format on

class WCSSTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
  protected:
    void SetUp() override { omp_set_num_threads(1); }

    static constexpr size_t N_SAMPLES = 10000;
    static constexpr size_t MAX_D = 768;
    static constexpr unsigned int SEED = 42;
    static constexpr int N_ITERS = 10;
    static constexpr float TOLERANCE = 0.01f;

    // Full test data loaded from disk (N_SAMPLES × MAX_D)
    static std::vector<float> full_data_;
    static void LoadTestData() {
        if (!full_data_.empty()) {
            return;
        }
        std::string data_file = CMAKE_SOURCE_DIR "/tests/test_data.bin";
        std::ifstream in(data_file, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Could not open test_data.bin. Run generate_wcss_ground_truth.out first.");
        }
        full_data_.resize(N_SAMPLES * MAX_D);
        in.read(reinterpret_cast<char*>(full_data_.data()), full_data_.size() * sizeof(float));
        in.close();
    }

    static std::vector<float> ExtractSubdim(size_t d) {
        LoadTestData();
        if (d > MAX_D) {
            throw std::runtime_error("Requested dimensionality exceeds MAX_D");
        }
        std::vector<float> data(N_SAMPLES * d);
        for (size_t i = 0; i < N_SAMPLES; ++i) {
            std::memcpy(&data[i * d], &full_data_[i * MAX_D], d * sizeof(float));
        }
        return data;
    }
};

std::vector<float> WCSSTest::full_data_;

/**
 * @brief Test that WCSS monotonically decreases across iterations
 * and matches expected ground truth values (within 10% tolerance).
 *
 */
TEST_P(WCSSTest, MonotonicallyDecreases_AndMatchesGroundTruth) {
    auto [n_clusters, d] = GetParam();

    if (n_clusters > N_SAMPLES) {
        GTEST_SKIP() << "Skipping: n_clusters (" << n_clusters << ") > n_samples (" << N_SAMPLES
                     << ")";
    }

    auto data = ExtractSubdim(d);

    ASSERT_EQ(data.size(), N_SAMPLES * d) << "Data size mismatch";

    // These values match the defaults at the time ground truth was generated
    skmeans::SuperKMeansConfig config;
    config.iters = N_ITERS;
    config.verbose = false;
    config.seed = SEED;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.min_not_pruned_pct = 0.03f;
    config.max_not_pruned_pct = 0.05f;
    config.adjustment_factor_for_partial_d = 0.20f;
    config.angular = false;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), N_SAMPLES);
    const auto& stats = kmeans.iteration_stats;
    ASSERT_GE(stats.size(), 1) << "Expected at least 1 iteration recorded";

    // WCSS should monotonically decrease (or stay same)
    for (size_t i = 1; i < stats.size(); ++i) {
        float prev_wcss = stats[i - 1].objective;
        float curr_wcss = stats[i].objective;

        float tolerance = prev_wcss * 1e-6f;
        EXPECT_LE(curr_wcss, prev_wcss + tolerance)
            << "WCSS increased at iteration " << (i + 1) << ": " << prev_wcss << " -> " << curr_wcss
            << " (n_clusters=" << n_clusters << ", d=" << d << ")";
    }

    // Final WCSS should match ground truth
    auto key = std::make_pair(n_clusters, d);
    auto it = GROUND_TRUTH.find(key);
    ASSERT_NE(it, GROUND_TRUTH.end()) << "No ground truth for k=" << n_clusters << ", d=" << d;

    float expected_wcss = it->second;
    float final_wcss = stats.back().objective;
    float wcss_upper_bound = expected_wcss * (1.0f + TOLERANCE);
    EXPECT_LE(final_wcss, wcss_upper_bound)
        << "WCSS too high (k=" << n_clusters << ", d=" << d << "): " << final_wcss << " > "
        << wcss_upper_bound << " (expected ~" << expected_wcss << ")";

    // WCSS shouldn't be drastically lower (would indicate a bug or wrong test setup)
    float wcss_lower_bound = expected_wcss * 0.5f;
    EXPECT_GE(final_wcss, wcss_lower_bound)
        << "WCSS suspiciously low (k=" << n_clusters << ", d=" << d << "): " << final_wcss << " < "
        << wcss_lower_bound << " (expected ~" << expected_wcss << ")";

    // All iteration stats should have valid values
    for (size_t i = 0; i < stats.size(); ++i) {
        EXPECT_EQ(stats[i].iteration, i + 1) << "Iteration number mismatch at index " << i;
        EXPECT_GT(stats[i].objective, 0.0f) << "WCSS should be positive at iteration " << (i + 1);
        EXPECT_TRUE(std::isfinite(stats[i].objective))
            << "WCSS is not finite at iteration " << (i + 1);
        EXPECT_TRUE(std::isfinite(stats[i].shift))
            << "Shift is not finite at iteration " << (i + 1);
    }

    // Verify centroids are valid
    EXPECT_EQ(centroids.size(), n_clusters * d) << "Centroid size mismatch";
    for (size_t i = 0; i < centroids.size(); ++i) {
        EXPECT_TRUE(std::isfinite(centroids[i])) << "Centroid value not finite at index " << i;
    }

    // Verify is_gemm_only flag is correct
    // When d < DIMENSION_THRESHOLD_FOR_PRUNING or n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING,
    // all iterations after the first should use GEMM-only (no pruning)
    bool should_use_gemm_only = (d < skmeans::DIMENSION_THRESHOLD_FOR_PRUNING) ||
                                (n_clusters <= skmeans::N_CLUSTERS_THRESHOLD_FOR_PRUNING);
    if (should_use_gemm_only && stats.size() > 1) {
        for (size_t i = 1; i < stats.size(); ++i) {
            EXPECT_TRUE(stats[i].is_gemm_only)
                << "Expected is_gemm_only=true for iteration " << (i + 1) << " (d=" << d << " < "
                << skmeans::DIMENSION_THRESHOLD_FOR_PRUNING << " or k=" << n_clusters
                << " <= " << skmeans::N_CLUSTERS_THRESHOLD_FOR_PRUNING << ")";
        }
    }
}

/**
 * @brief Test that WCSS with BLAS-only mode (no pruning) monotonically decreases
 * and matches expected ground truth values (within 1% tolerance).
 *
 */
TEST_P(WCSSTest, BlasOnly_MonotonicallyDecreases_AndMatchesGroundTruth) {
    auto [n_clusters, d] = GetParam();

    // Only run this test for specific dimensions; otherwise test is too long
    const std::vector<size_t> allowed_dims = {384, 512, 600, 768};
    if (std::find(allowed_dims.begin(), allowed_dims.end(), d) == allowed_dims.end()) {
        SUCCEED();
        return;
    }

    if (n_clusters > N_SAMPLES) {
        SUCCEED();
        return;
    }

    auto data = ExtractSubdim(d);

    ASSERT_EQ(data.size(), N_SAMPLES * d) << "Data size mismatch";

    // Same config as default test, but with use_blas_only=true
    skmeans::SuperKMeansConfig config;
    config.iters = N_ITERS;
    config.verbose = false;
    config.seed = SEED;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.min_not_pruned_pct = 0.03f;
    config.max_not_pruned_pct = 0.05f;
    config.adjustment_factor_for_partial_d = 0.20f;
    config.angular = false;
    config.use_blas_only = true; // Enable BLAS-only mode (no pruning)

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), N_SAMPLES);
    const auto& stats = kmeans.iteration_stats;
    ASSERT_GE(stats.size(), 1) << "Expected at least 1 iteration recorded";

    for (size_t i = 1; i < stats.size(); ++i) {
        float prev_wcss = stats[i - 1].objective;
        float curr_wcss = stats[i].objective;
        float tolerance = prev_wcss * 1e-6f;
        EXPECT_LE(curr_wcss, prev_wcss + tolerance)
            << "WCSS increased at iteration " << (i + 1) << " [BLAS-only mode]: " << prev_wcss
            << " -> " << curr_wcss << " (n_clusters=" << n_clusters << ", d=" << d << ")";
    }

    auto key = std::make_pair(n_clusters, d);
    auto it = GROUND_TRUTH.find(key);
    ASSERT_NE(it, GROUND_TRUTH.end()) << "No ground truth for k=" << n_clusters << ", d=" << d;

    float expected_wcss = it->second;
    float final_wcss = stats.back().objective;
    float wcss_upper_bound = expected_wcss * (1.0f + TOLERANCE);
    EXPECT_LE(final_wcss, wcss_upper_bound)
        << "WCSS too high [BLAS-only mode] (k=" << n_clusters << ", d=" << d << "): " << final_wcss
        << " > " << wcss_upper_bound << " (expected ~" << expected_wcss << ")";

    // WCSS shouldn't be drastically lower
    float wcss_lower_bound = expected_wcss * 0.5f;
    EXPECT_GE(final_wcss, wcss_lower_bound)
        << "WCSS suspiciously low [BLAS-only mode] (k=" << n_clusters << ", d=" << d
        << "): " << final_wcss << " < " << wcss_lower_bound << " (expected ~" << expected_wcss
        << ")";

    // All iteration stats should have valid values
    for (size_t i = 0; i < stats.size(); ++i) {
        EXPECT_EQ(stats[i].iteration, i + 1) << "Iteration number mismatch at index " << i;
        EXPECT_GT(stats[i].objective, 0.0f) << "WCSS should be positive at iteration " << (i + 1);
        EXPECT_TRUE(std::isfinite(stats[i].objective))
            << "WCSS is not finite at iteration " << (i + 1);
        EXPECT_TRUE(std::isfinite(stats[i].shift))
            << "Shift is not finite at iteration " << (i + 1);
    }

    // Verify centroids are valid
    EXPECT_EQ(centroids.size(), n_clusters * d) << "Centroid size mismatch";
    for (size_t i = 0; i < centroids.size(); ++i) {
        EXPECT_TRUE(std::isfinite(centroids[i])) << "Centroid value not finite at index " << i;
    }

    // Verify that all iterations after the first use GEMM-only when use_blas_only=true
    // (First iteration uses FirstAssignAndUpdateCentroids which doesn't set is_gemm_only)
    if (stats.size() > 1) {
        for (size_t i = 1; i < stats.size(); ++i) {
            EXPECT_TRUE(stats[i].is_gemm_only)
                << "Expected is_gemm_only=true for iteration " << (i + 1)
                << " when use_blas_only=true (k=" << n_clusters << ", d=" << d << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    WCSSParameterized,
    WCSSTest,
    ::testing::Combine(
        ::testing::Values(10, 100, 250),
        ::testing::Values(4, 16, 32, 64, 100, 128, 384, 512, 600, 768)
    ),
    [](const ::testing::TestParamInfo<WCSSTest::ParamType>& info) {
        return "k" + std::to_string(std::get<0>(info.param)) + "_d" +
               std::to_string(std::get<1>(info.param));
    }
);

} // anonymous namespace
