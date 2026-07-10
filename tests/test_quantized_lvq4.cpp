#undef HAS_FFTW

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/quantizers/quantizer.h"
#include "superkmeans/quantizers/lvq4.h"
#include "superkmeans/superkmeans.h"

#include "recall_utils.h"

using namespace skmeans;

namespace {

using skm_u8 = SuperKMeans<Quantization::u8, DistanceFunction::l2>;

} // namespace

// ── LVQ4Quantizer unit tests ──

class LVQ4QuantizerTest : public ::testing::Test {
  protected:
    static constexpr size_t n = 1000;
    static constexpr size_t d = 128;
    std::vector<float> data;
    float data_min = 0.0f, data_max = 0.0f;

    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n * d);
        data_min = std::numeric_limits<float>::max();
        data_max = std::numeric_limits<float>::lowest();
        for (auto& v : data) {
            v = dist(rng);
            data_min = std::min(data_min, v);
            data_max = std::max(data_max, v);
        }
    }
};

TEST_F(LVQ4QuantizerTest, FitEncodeDecode_Roundtrip) {
    LVQ4Quantizer quantizer;
    EXPECT_FALSE(quantizer.IsFitted());
    quantizer.Fit(data.data(), n, d);
    EXPECT_TRUE(quantizer.IsFitted());

    const size_t cs = quantizer.CodeSize(d);
    EXPECT_EQ(cs, d / 2 + 8);

    std::vector<uint8_t> encoded(n * cs);
    quantizer.Encode(data.data(), encoded.data(), n, d);
    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded.data(), decoded.data(), n, d);

    // 4-bit per-vector quantization: error bounded by the per-vector step,
    // which is at most global_range / 15.
    const float max_err = (data_max - data_min) / 15.0f;
    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_NEAR(data[i], decoded[i], max_err + 1e-4f) << "at index " << i;
    }
}

TEST_F(LVQ4QuantizerTest, OddDimensionality_Throws) {
    std::vector<float> tiny(63, 0.1f);
    EXPECT_THROW(LVQ4Quantizer().Fit(tiny.data(), 1, 63), std::invalid_argument);
    // Even dimensionality must not throw.
    EXPECT_NO_THROW(LVQ4Quantizer().Fit(tiny.data(), 1, 62));
}

TEST_F(LVQ4QuantizerTest, EncodedNibbles_InRange) {
    LVQ4Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);
    const size_t cs = quantizer.CodeSize(d);
    const size_t nibble_bytes = d / 2;

    std::vector<uint8_t> encoded(n * cs);
    quantizer.Encode(data.data(), encoded.data(), n, d);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* code = encoded.data() + i * cs;
        for (size_t b = 0; b < nibble_bytes; ++b) {
            EXPECT_LE(code[b] & 0x0F, 15u);
            EXPECT_LE(code[b] >> 4, 15u);
        }
    }
}

TEST_F(LVQ4QuantizerTest, FindNearestNeighbor_MatchesBruteForce) {
    LVQ4Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);
    const size_t cs = quantizer.CodeSize(d);

    size_t n_centroids = 50;
    size_t n_queries = 200;

    std::vector<uint8_t> codes(n * cs);
    quantizer.Encode(data.data(), codes.data(), n, d);
    std::vector<float> norms(n);
    quantizer.ComputeNorms(codes.data(), n, d, norms.data());

    const uint8_t* queries = codes.data() + n_centroids * cs;
    const uint8_t* centroids = codes.data();

    std::vector<uint32_t> knn(n_queries);
    std::vector<float> distances(n_queries);
    std::vector<float> tmp_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

    quantizer.FindNearestNeighbor(
        queries, centroids, data.data() + n_centroids * d, data.data(),
        n_queries, n_centroids, d,
        norms.data() + n_centroids, norms.data(),
        knn.data(), distances.data(), tmp_buf.data()
    );

    // LVQ4's distance is exact L2 on the decoded vectors, so the returned
    // distance must match the brute-force minimum (up to float rounding).
    std::vector<float> decoded(n * d);
    quantizer.Decode(codes.data(), decoded.data(), n, d);
    for (size_t i = 0; i < n_queries; ++i) {
        float bf_min = std::numeric_limits<float>::max();
        for (size_t j = 0; j < n_centroids; ++j) {
            float dist = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                float diff = decoded[(n_centroids + i) * d + k] - decoded[j * d + k];
                dist += diff * diff;
            }
            bf_min = std::min(bf_min, dist);
        }
        ASSERT_LT(knn[i], n_centroids);
        EXPECT_LE(distances[i], bf_min + std::max(1e-2f, bf_min * 1e-3f)) << "query " << i;
    }
}

// ── SuperKMeans<u8> integration tests (LVQ4) ──

class SuperKMeansLVQ4Test : public ::testing::Test {};

TEST_F(SuperKMeansLVQ4Test, BasicTraining) {
    const size_t n = 2000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::lvq4;

    auto kmeans = skm_u8(n_clusters, d, config);
    EXPECT_FALSE(kmeans.IsTrained());
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), n_clusters * d);
    EXPECT_EQ(kmeans.GetNClusters(), n_clusters);
}

TEST_F(SuperKMeansLVQ4Test, AllClustersUsed_AssignmentsValid) {
    const size_t n = 5000, d = 128, n_clusters = 20;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 15;
    config.quantizer_type = QuantizerType::lvq4;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    EXPECT_EQ(assignments.size(), n);
    for (size_t i = 0; i < n; ++i) EXPECT_LT(assignments[i], n_clusters) << "at " << i;

    std::unordered_set<uint32_t> used(assignments.begin(), assignments.end());
    EXPECT_EQ(used.size(), n_clusters);
}

TEST_F(SuperKMeansLVQ4Test, Recall_MatchesGroundTruth) {
    float recall = skm_test::ClusteringRecall<Quantization::u8>(
        QuantizerType::lvq4, CMAKE_SOURCE_DIR "/tests/test_data.bin"
    );
    EXPECT_NEAR(recall, skm_test::RECALL_GROUND_TRUTH.at("lvq4"), skm_test::RECALL_TOL);
}

TEST_F(SuperKMeansLVQ4Test, QuantizedCentroidUpdate_Converges) {
    const size_t n = 3000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::lvq4;
    config.quantized_centroid_update = true;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_EQ(centroids.size(), n_clusters * d);

    auto stats = kmeans.GetIterationStats();
    ASSERT_GE(stats.size(), 2u);
    EXPECT_LT(stats.back().objective, stats.front().objective);
}

TEST_F(SuperKMeansLVQ4Test, QuantizedAssign_MatchesAssign) {
    const size_t n = 3000, d = 64, n_clusters = 12;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::lvq4;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    auto assign_gt = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);
    auto assign_q = kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);

    size_t agree = 0;
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(assign_q[i], n_clusters);
        if (assign_gt[i] == assign_q[i]) agree++;
    }
    EXPECT_GT(static_cast<double>(agree) / n, 0.5);
}

TEST_F(SuperKMeansLVQ4Test, QuantizedAssign_RepeatedDifferentData) {
    const size_t n = 3000, d = 64, n_clusters = 12;
    std::vector<float> dataA = MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 1);
    std::vector<float> dataB = MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 999);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::lvq4;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(dataA.data(), n);

    (void) kmeans.QuantizedAssign(dataA.data(), centroids.data(), n, n_clusters);
    auto q_B = kmeans.QuantizedAssign(dataB.data(), centroids.data(), n, n_clusters);
    auto gt_B = kmeans.Assign(dataB.data(), centroids.data(), n, n_clusters);

    size_t agree = 0;
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(q_B[i], n_clusters);
        if (q_B[i] == gt_B[i]) agree++;
    }
    EXPECT_GT(static_cast<double>(agree) / n, 0.5)
        << "QuantizedAssign did not reflect the second (different) dataset";
}

TEST_F(SuperKMeansLVQ4Test, InvalidInputs_Throw) {
    const size_t n = 10000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);
    auto make = [&](const SuperKMeansConfig& c, size_t k, size_t dd) { return skm_u8(k, dd, c); };
    SuperKMeansConfig base;
    base.quantizer_type = QuantizerType::lvq4;

    EXPECT_THROW(([&]{ auto km = make(base, n + 10, d); km.Train(data.data(), n); }()),
                 std::runtime_error);
    EXPECT_THROW(([&]{
        SuperKMeansConfig c = base; c.sampling_fraction = 0.0001f; c.max_points_per_cluster = 1;
        auto km = make(c, n_clusters, d); km.Train(data.data(), n);
    }()), std::runtime_error);
    EXPECT_THROW(make(base, 0, d), std::invalid_argument);
    EXPECT_THROW(make(base, n_clusters, 0), std::invalid_argument);
    EXPECT_THROW(([&]{ SuperKMeansConfig c = base; c.iters = 0; make(c, n_clusters, d); }()),
                 std::invalid_argument);
    EXPECT_THROW(([&]{ SuperKMeansConfig c = base; c.sampling_fraction = 0.0f; make(c, n_clusters, d); }()),
                 std::invalid_argument);
    EXPECT_THROW(([&]{ SuperKMeansConfig c = base; c.sampling_fraction = 1.5f; make(c, n_clusters, d); }()),
                 std::invalid_argument);
    EXPECT_THROW(([&]{
        auto km = make(base, n_clusters, d);
        km.Train(data.data(), n);
        km.Train(data.data(), n);
    }()), std::runtime_error);
}

TEST_F(SuperKMeansLVQ4Test, EarlyTermination) {
    const size_t n = 10000, d = 64, n_clusters = 5, max_iters = 100;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, false, 0.5f, 20.0f);

    SuperKMeansConfig c_early;
    c_early.iters = max_iters;
    c_early.early_termination = true;
    c_early.tol = 1e-2f;
    c_early.sampling_fraction = 1.0f;
    c_early.quantizer_type = QuantizerType::lvq4;
    auto km_early = skm_u8(n_clusters, d, c_early);
    km_early.Train(data.data(), n);
    size_t iters_early = km_early.iteration_stats.size();

    SuperKMeansConfig c_no;
    c_no.iters = max_iters;
    c_no.early_termination = false;
    c_no.sampling_fraction = 1.0f;
    c_no.quantizer_type = QuantizerType::lvq4;
    auto km_no = skm_u8(n_clusters, d, c_no);
    km_no.Train(data.data(), n);
    size_t iters_no = km_no.iteration_stats.size();

    EXPECT_LT(iters_early, max_iters);
    EXPECT_EQ(iters_no, max_iters);
    EXPECT_LT(iters_early, iters_no);
}

TEST_F(SuperKMeansLVQ4Test, AngularMode_NormalizesCentroids) {
    const size_t n = 5000, d = 64, n_clusters = 50;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.angular = true;
    config.quantizer_type = QuantizerType::lvq4;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    for (size_t c = 0; c < n_clusters; ++c) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) norm += centroids[c * d + j] * centroids[c * d + j];
        EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-4f) << "centroid " << c;
    }
}

TEST_F(SuperKMeansLVQ4Test, Determinism_SameSeedSameCentroids) {
    const size_t n = 3000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.early_termination = false;
    config.seed = 123;
    config.n_threads = 1;
    config.quantizer_type = QuantizerType::lvq4;

    auto c1 = skm_u8(n_clusters, d, config).Train(data.data(), n);
    auto c2 = skm_u8(n_clusters, d, config).Train(data.data(), n);
    ASSERT_EQ(c1.size(), c2.size());
    for (size_t i = 0; i < c1.size(); ++i) EXPECT_FLOAT_EQ(c1[i], c2[i]) << "at " << i;
}

TEST_F(SuperKMeansLVQ4Test, SmallestEvenDimension) {
    const size_t n = 1000, d = 2, n_clusters = 8;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::lvq4;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_EQ(centroids.size(), n_clusters * d);
}

// ── Pruning integration tests (need d >= 128 and k > 256) ──

class SuperKMeansLVQ4PruningTest : public ::testing::Test {};

TEST_F(SuperKMeansLVQ4PruningTest, SLOW_PruningConverges) {
    const size_t n = 10000, d = 128, n_clusters = 300;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 5;
    config.quantizer_type = QuantizerType::lvq4;
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

TEST_F(SuperKMeansLVQ4PruningTest, SLOW_PruningRecallCloseToNoPruning) {
    const std::string path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    float pruned = skm_test::ClusteringRecall<Quantization::u8>(QuantizerType::lvq4, path, 300, 1024, true);
    float unpruned = skm_test::ClusteringRecall<Quantization::u8>(QuantizerType::lvq4, path, 300, 1024, false);
    EXPECT_NEAR(pruned, unpruned, skm_test::RECALL_PRUNE_TOL)
        << "pruned=" << pruned << " unpruned=" << unpruned;
}

TEST_F(SuperKMeansLVQ4PruningTest, SLOW_AssignTrainingPointsReuseMatchesQuantizedAssign) {
    const size_t n = 5000, d = 128, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::lvq4;
    config.sampling_fraction = 1.0f;
    config.use_blas_only = false;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);

    auto approximate_assignments = kmeans.AssignTrainingPoints(data.data(), centroids.data(), n, n_clusters);
    auto brute_force_assignments = kmeans.QuantizedAssign(data.data(), centroids.data(), n, n_clusters);

    ASSERT_EQ(approximate_assignments.size(), n);
    size_t agree = 0;
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(approximate_assignments[i], n_clusters);
        if (approximate_assignments[i] == brute_force_assignments[i]) agree++;
    }
    const double ratio = static_cast<double>(agree) / n;
    EXPECT_GT(ratio, 0.98) << "agreement=" << ratio;
}
