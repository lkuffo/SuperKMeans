#include <algorithm>
#include <cmath>
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

// Ground truth WCSS values for each (n_clusters, dimensionality) combination
// Generated with: N_SAMPLES=100000, N_TRUE_CENTERS=100, CLUSTER_STD=1.0, CENTER_SPREAD=10.0,
// SEED=42, N_ITERS=10 Regenerate with: ./generate_wcss_ground_truth.out Format: {n_clusters,
// dimensionality} -> expected_wcss

// clang-format off
const std::map<std::pair<size_t, size_t>, float> GROUND_TRUTH = {
    // k=10
    {{10, 4}, 1.45082e+07f},
    {{10, 16}, 1.15181e+08f},
    {{10, 32}, 2.55024e+08f},
    {{10, 64}, 5.65870e+08f},
    {{10, 100}, 8.69263e+08f},
    {{10, 128}, 1.12330e+09f},
    {{10, 384}, 3.45538e+09f},
    {{10, 512}, 4.60039e+09f},
    {{10, 600}, 5.43918e+09f},
    {{10, 768}, 6.96413e+09f},
    {{10, 900}, 8.17673e+09f},
    {{10, 1024}, 9.25543e+09f},
    {{10, 1536}, 1.38387e+10f},
    {{10, 2000}, 1.81425e+10f},
    {{10, 3072}, 2.78603e+10f},
    // k=100
    {{100, 4}, 1.08799e+06f},
    {{100, 16}, 1.84771e+07f},
    {{100, 32}, 4.45723e+07f},
    {{100, 64}, 1.50841e+08f},
    {{100, 100}, 2.21559e+08f},
    {{100, 128}, 2.96399e+08f},
    {{100, 384}, 9.62221e+08f},
    {{100, 512}, 1.35008e+09f},
    {{100, 600}, 1.52428e+09f},
    {{100, 768}, 1.91229e+09f},
    {{100, 900}, 2.14949e+09f},
    {{100, 1024}, 2.95718e+09f},
    {{100, 1536}, 4.23874e+09f},
    {{100, 2000}, 5.24159e+09f},
    {{100, 3072}, 7.54917e+09f},
    // k=1000
    {{1000, 4}, 1.69094e+05f},
    {{1000, 16}, 1.28640e+06f},
    {{1000, 32}, 2.83718e+06f},
    {{1000, 64}, 5.97009e+06f},
    {{1000, 100}, 9.50948e+06f},
    {{1000, 128}, 1.22841e+07f},
    {{1000, 384}, 3.76193e+07f},
    {{1000, 512}, 5.03048e+07f},
    {{1000, 600}, 5.90127e+07f},
    {{1000, 768}, 7.56619e+07f},
    {{1000, 900}, 8.87550e+07f},
    {{1000, 1024}, 1.01012e+08f},
    {{1000, 1536}, 1.51711e+08f},
    {{1000, 2000}, 1.97681e+08f},
    {{1000, 3072}, 3.03797e+08f},
    // k=10000
    {{10000, 4}, 4.54627e+04f},
    {{10000, 16}, 8.20825e+05f},
    {{10000, 32}, 2.14894e+06f},
    {{10000, 64}, 4.94882e+06f},
    {{10000, 100}, 8.14074e+06f},
    {{10000, 128}, 1.06430e+07f},
    {{10000, 384}, 3.34897e+07f},
    {{10000, 512}, 4.49184e+07f},
    {{10000, 600}, 5.27915e+07f},
    {{10000, 768}, 6.78171e+07f},
    {{10000, 900}, 7.96324e+07f},
    {{10000, 1024}, 9.07352e+07f},
    {{10000, 1536}, 1.36584e+08f},
    {{10000, 2000}, 1.78188e+08f},
    {{10000, 3072}, 2.74325e+08f},
};
// clang-format on

class WCSSTest : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {
  protected:
    void SetUp() override { omp_set_num_threads(omp_get_max_threads()); }

    static constexpr size_t N_SAMPLES = 100000;
    static constexpr size_t N_TRUE_CENTERS = 100;
    static constexpr float CLUSTER_STD = 1.0f;
    static constexpr float CENTER_SPREAD = 10.0f;
    static constexpr unsigned int SEED = 42;
    static constexpr int N_ITERS = 10;
    static constexpr float TOLERANCE = 0.07f;
};

/**
 * @brief Test that WCSS monotonically decreases across iterations
 * and matches expected ground truth values (within 7% tolerance).
 *
 */
TEST_P(WCSSTest, MonotonicallyDecreases_AndMatchesGroundTruth) {
    auto [n_clusters, d] = GetParam();

    if (n_clusters > N_SAMPLES) {
        GTEST_SKIP() << "Skipping: n_clusters (" << n_clusters << ") > n_samples (" << N_SAMPLES
                     << ")";
    }

    std::vector<float> data =
        skmeans::MakeBlobs(N_SAMPLES, d, N_TRUE_CENTERS, false, CLUSTER_STD, CENTER_SPREAD, SEED);

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

    std::vector<float> data =
        skmeans::MakeBlobs(N_SAMPLES, d, N_TRUE_CENTERS, false, CLUSTER_STD, CENTER_SPREAD, SEED);

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
        ::testing::Values(10, 100, 1000),
        ::testing::Values(4, 16, 32, 64, 100, 128, 384, 512, 600, 768, 900, 1024, 1536, 2000, 3072)
    ),
    [](const ::testing::TestParamInfo<WCSSTest::ParamType>& info) {
        return "k" + std::to_string(std::get<0>(info.param)) + "_d" +
               std::to_string(std::get<1>(info.param));
    }
);

} // anonymous namespace
