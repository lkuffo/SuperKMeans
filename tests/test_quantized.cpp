#undef HAS_FFTW

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

#include "recall_utils.h"

using namespace skmeans;

namespace {

using skm_u8 = SuperKMeans<Quantization::u8, DistanceFunction::l2>;

struct QParam {
    QuantizerType type;
    const char* name;
};

} // namespace

class QuantizedIntegrationTest : public ::testing::TestWithParam<QParam> {};

INSTANTIATE_TEST_SUITE_P(
    Quantizers,
    QuantizedIntegrationTest,
    ::testing::Values(
        QParam{QuantizerType::sq8, "sq8"},
        QParam{QuantizerType::lvq4, "lvq4"},
        QParam{QuantizerType::rabitq, "rabitq"}
    ),
    [](const ::testing::TestParamInfo<QParam>& info) { return std::string(info.param.name); }
);

TEST_P(QuantizedIntegrationTest, BasicTraining) {
    const size_t n = 2000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = GetParam().type;

    auto kmeans = skm_u8(n_clusters, d, config);
    EXPECT_FALSE(kmeans.IsTrained());
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), n_clusters * d);
    EXPECT_EQ(kmeans.GetNClusters(), n_clusters);
}

TEST_P(QuantizedIntegrationTest, AllClustersUsed_AssignmentsValid) {
    const size_t n = 5000, d = 128, n_clusters = 20;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 15;
    config.quantizer_type = GetParam().type;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    EXPECT_EQ(assignments.size(), n);
    for (size_t i = 0; i < n; ++i)
        EXPECT_LT(assignments[i], n_clusters) << "at " << i;

    std::unordered_set<uint32_t> used(assignments.begin(), assignments.end());
    EXPECT_EQ(used.size(), n_clusters);
}

TEST_P(QuantizedIntegrationTest, Recall_MatchesGroundTruth) {
    float recall = skm_test::ClusteringRecall<Quantization::u8>(
        GetParam().type, CMAKE_SOURCE_DIR "/tests/test_data.bin"
    );
    EXPECT_NEAR(recall, skm_test::RECALL_GROUND_TRUTH.at(GetParam().name), skm_test::RECALL_TOL);
}

TEST_P(QuantizedIntegrationTest, QuantizedCentroidUpdate_Converges) {
    const size_t n = 3000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = GetParam().type;
    config.quantized_centroid_update = true;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_EQ(centroids.size(), n_clusters * d);

    auto stats = kmeans.GetIterationStats();
    ASSERT_GE(stats.size(), 2u);
    EXPECT_LT(stats.back().objective, stats.front().objective);
}

TEST_P(QuantizedIntegrationTest, QuantizedAssign_MatchesAssign) {
    const size_t n = 3000, d = 128, n_clusters = 15;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 5;
    config.quantizer_type = GetParam().type;
    config.sampling_fraction = 1.0f;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    auto assign_gt = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    auto assign_q = kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);

    std::vector<size_t> sizes(n_clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(assign_q[i], n_clusters) << "at " << i;
        sizes[assign_q[i]]++;
    }
    size_t max_size = *std::max_element(sizes.begin(), sizes.end());
    EXPECT_LT(max_size, n / 2) << "imbalanced (max cluster " << max_size << ")";

    size_t agree = 0;
    for (size_t i = 0; i < n; ++i)
        if (assign_gt[i] == assign_q[i])
            agree++;
    EXPECT_GT(static_cast<double>(agree) / n, 0.5);
}

TEST_P(QuantizedIntegrationTest, QuantizedAssign_RepeatedDifferentData) {
    const size_t n = 3000, d = 64, n_clusters = 12;
    std::vector<float> dataA = MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 1);
    std::vector<float> dataB = MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 999);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = GetParam().type;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(dataA.data(), n);

    (void) kmeans.QuantizedAssign(dataA.data(), centroids.data(), n, n_clusters);
    auto q_B = kmeans.QuantizedAssign(dataB.data(), centroids.data(), n, n_clusters);
    auto gt_B = kmeans.Assign(dataB.data(), centroids.data(), n, n_clusters);

    size_t agree = 0;
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(q_B[i], n_clusters);
        if (q_B[i] == gt_B[i])
            agree++;
    }
    EXPECT_GT(static_cast<double>(agree) / n, 0.5)
        << "QuantizedAssign did not reflect the second (different) dataset";
}

TEST(QuantizedConstruction, RequiresQuantizerType) {
    SuperKMeansConfig config;
    EXPECT_THROW(skm_u8(10, 64, config), std::invalid_argument);
}

TEST_P(QuantizedIntegrationTest, InvalidInputs_Throw) {
    const size_t n = 10000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);
    auto make = [&](const SuperKMeansConfig& c, size_t k, size_t dd) { return skm_u8(k, dd, c); };
    SuperKMeansConfig base;
    base.quantizer_type = GetParam().type;

    EXPECT_THROW(
        ([&] {
            auto km = make(base, n + 10, d);
            km.Train(data.data(), n);
        }()),
        std::runtime_error
    );
    EXPECT_THROW(
        ([&] {
            SuperKMeansConfig c = base;
            c.sampling_fraction = 0.0001f;
            c.max_points_per_cluster = 1;
            auto km = make(c, n_clusters, d);
            km.Train(data.data(), n);
        }()),
        std::runtime_error
    );
    EXPECT_THROW(make(base, 0, d), std::invalid_argument);
    EXPECT_THROW(make(base, n_clusters, 0), std::invalid_argument);
    EXPECT_THROW(
        ([&] {
            SuperKMeansConfig c = base;
            c.iters = 0;
            make(c, n_clusters, d);
        }()),
        std::invalid_argument
    );
    EXPECT_THROW(
        ([&] {
            SuperKMeansConfig c = base;
            c.sampling_fraction = 0.0f;
            make(c, n_clusters, d);
        }()),
        std::invalid_argument
    );
    EXPECT_THROW(
        ([&] {
            SuperKMeansConfig c = base;
            c.sampling_fraction = 1.5f;
            make(c, n_clusters, d);
        }()),
        std::invalid_argument
    );
    EXPECT_THROW(
        ([&] {
            auto km = make(base, n_clusters, d);
            km.Train(data.data(), n);
            km.Train(data.data(), n);
        }()),
        std::runtime_error
    );
}

TEST_P(QuantizedIntegrationTest, EarlyTermination) {
    const size_t n = 10000, d = 64, n_clusters = 5, max_iters = 100;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, false, 0.5f, 20.0f);

    SuperKMeansConfig c_early;
    c_early.iters = max_iters;
    c_early.early_termination = true;
    c_early.tol = 1e-2f;
    c_early.sampling_fraction = 1.0f;
    c_early.quantizer_type = GetParam().type;
    auto km_early = skm_u8(n_clusters, d, c_early);
    km_early.Train(data.data(), n);
    size_t iters_early = km_early.iteration_stats.size();

    SuperKMeansConfig c_no;
    c_no.iters = max_iters;
    c_no.early_termination = false;
    c_no.sampling_fraction = 1.0f;
    c_no.quantizer_type = GetParam().type;
    auto km_no = skm_u8(n_clusters, d, c_no);
    km_no.Train(data.data(), n);
    size_t iters_no = km_no.iteration_stats.size();

    EXPECT_LT(iters_early, max_iters);
    EXPECT_EQ(iters_no, max_iters);
    EXPECT_LT(iters_early, iters_no);
}

TEST_P(QuantizedIntegrationTest, AngularMode_NormalizesCentroids) {
    const size_t n = 5000, d = 64, n_clusters = 50;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.angular = true;
    config.quantizer_type = GetParam().type;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    for (size_t c = 0; c < n_clusters; ++c) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j)
            norm += centroids[c * d + j] * centroids[c * d + j];
        EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-4f) << "centroid " << c;
    }
}

TEST_P(QuantizedIntegrationTest, Determinism_SameSeedSameCentroids) {
    const size_t n = 3000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.early_termination = false;
    config.seed = 123;
    config.n_threads = 1;
    config.quantizer_type = GetParam().type;

    auto c1 = skm_u8(n_clusters, d, config).Train(data.data(), n);
    auto c2 = skm_u8(n_clusters, d, config).Train(data.data(), n);
    ASSERT_EQ(c1.size(), c2.size());
    for (size_t i = 0; i < c1.size(); ++i)
        EXPECT_FLOAT_EQ(c1[i], c2[i]) << "at " << i;
}

TEST_P(QuantizedIntegrationTest, FullPrecisionFinalCentroids_ImprovesReconstruction) {
    const size_t n = 5000, d = 64, n_clusters = 20;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    auto make_config = [&](bool full_precision) {
        SuperKMeansConfig config;
        config.iters = 10;
        config.seed = 42;
        config.n_threads = 1;
        config.sampling_fraction = 1.0f;
        config.early_termination = false;
        config.quantizer_type = GetParam().type;
        config.quantized_centroid_update = true;
        config.full_precision_final_centroids = full_precision;
        return config;
    };

    auto centroids_q = skm_u8(n_clusters, d, make_config(false)).Train(data.data(), n);
    auto centroids_fp = skm_u8(n_clusters, d, make_config(true)).Train(data.data(), n);

    ASSERT_EQ(centroids_q.size(), n_clusters * d);
    ASSERT_EQ(centroids_fp.size(), n_clusters * d);
    for (float v : centroids_fp)
        ASSERT_TRUE(std::isfinite(v));

    float max_diff = 0.0f;
    for (size_t i = 0; i < centroids_q.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(centroids_q[i] - centroids_fp[i]));
    }
    EXPECT_GT(max_diff, 1e-4f) << "full_precision_final_centroids did not change the centroids";

    skm_u8 assigner(n_clusters, d, make_config(false));
    auto a_q = assigner.Assign(data.data(), centroids_q.data(), n, n_clusters);
    auto a_fp = assigner.Assign(data.data(), centroids_fp.data(), n, n_clusters);
    double wcss_q = skm_u8::ComputeWCSS(data.data(), centroids_q.data(), a_q.data(), n, d);
    double wcss_fp = skm_u8::ComputeWCSS(data.data(), centroids_fp.data(), a_fp.data(), n, d);
    EXPECT_LE(wcss_fp, wcss_q) << "full-precision WCSS " << wcss_fp << " > quantized WCSS "
                               << wcss_q;
}

// Pruning integration tests (need d >= 128 and k > 256)

TEST_P(QuantizedIntegrationTest, SLOW_PruningConverges) {
    const size_t n = 10000, d = 128, n_clusters = 300;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 5;
    config.quantizer_type = GetParam().type;
    config.use_blas_only = false;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_EQ(centroids.size(), n_clusters * d);

    auto stats = kmeans.GetIterationStats();
    ASSERT_GE(stats.size(), 2u);
    EXPECT_TRUE(stats[0].is_gemm_only);
    EXPECT_FALSE(stats[1].is_gemm_only);
    EXPECT_LT(stats.back().objective, stats.front().objective);
}

TEST_P(QuantizedIntegrationTest, SLOW_PruningRecallCloseToNoPruning) {
    const std::string path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    float pruned =
        skm_test::ClusteringRecall<Quantization::u8>(GetParam().type, path, 300, 1024, true);
    float unpruned =
        skm_test::ClusteringRecall<Quantization::u8>(GetParam().type, path, 300, 1024, false);
    EXPECT_NEAR(pruned, unpruned, skm_test::RECALL_PRUNE_TOL)
        << "pruned=" << pruned << " unpruned=" << unpruned;
}

TEST_P(QuantizedIntegrationTest, SLOW_AssignTrainingPointsRecallMatchesQuantizedAssign) {
    const std::string path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    auto r =
        skm_test::ClusteringRecallAssignMethods<Quantization::u8>(GetParam().type, path, 300, 1024);
    EXPECT_GT(r.assign_training_points, 0.0f);
    EXPECT_GT(r.quantized_assign, 0.0f);
    EXPECT_NEAR(r.assign_training_points, r.quantized_assign, skm_test::RECALL_PRUNE_TOL)
        << "atp=" << r.assign_training_points << " qa=" << r.quantized_assign;
}

TEST_P(QuantizedIntegrationTest, SLOW_AssignTrainingPointsReuseMatchesQuantizedAssign) {
    if (GetParam().type != QuantizerType::rabitq) {
        GTEST_SKIP() << "reuse and QuantizedAssign only share a quantization domain for rabitq";
    }
    const size_t n = 5000, d = 128, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = GetParam().type;
    config.sampling_fraction = 1.0f;
    config.use_blas_only = false;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    auto approximate_assignments =
        kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);
    auto brute_force_assignments =
        kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);

    ASSERT_EQ(approximate_assignments.size(), n);
    size_t agree = 0;
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(approximate_assignments[i], n_clusters);
        if (approximate_assignments[i] == brute_force_assignments[i])
            agree++;
    }
    const double ratio = static_cast<double>(agree) / n;
    EXPECT_GT(ratio, 0.98) << "agreement=" << ratio;
}
