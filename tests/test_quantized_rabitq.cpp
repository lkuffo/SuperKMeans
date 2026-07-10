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
#include "superkmeans/quantizers/rabitq.h"
#include "superkmeans/superkmeans.h"

#include "recall_utils.h"

using namespace skmeans;

namespace {

using skm_u8 = SuperKMeans<Quantization::u8, DistanceFunction::l2>;

std::vector<float> ColumnMean(const std::vector<float>& data, size_t n, size_t d) {
    std::vector<double> acc(d, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) acc[j] += data[i * d + j];
    std::vector<float> mean(d);
    for (size_t j = 0; j < d; ++j) mean[j] = static_cast<float>(acc[j] / static_cast<double>(n));
    return mean;
}

// Well-separated clusters: queries are tight around distinct, far-apart centroids,
// so the true nearest centroid is unambiguous even for a 1-bit code.
struct SeparatedSet {
    std::vector<float> centroids; // n_c * d
    std::vector<float> queries;   // n_q * d
    std::vector<uint32_t> true_nn;
    size_t n_c, n_q, d;
};

SeparatedSet MakeSeparated(size_t d, size_t n_c, size_t per, uint32_t seed) {
    SeparatedSet s;
    s.d = d; s.n_c = n_c; s.n_q = n_c * per;
    std::mt19937 rng(seed);
    std::normal_distribution<float> cdist(0.0f, 20.0f);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    s.centroids.resize(n_c * d);
    for (auto& v : s.centroids) v = cdist(rng);
    s.queries.resize(s.n_q * d);
    s.true_nn.resize(s.n_q);
    for (size_t i = 0; i < s.n_q; ++i) {
        uint32_t c = static_cast<uint32_t>(i % n_c);
        s.true_nn[i] = c;
        for (size_t j = 0; j < d; ++j) s.queries[i * d + j] = s.centroids[c * d + j] + noise(rng);
    }
    return s;
}

double NearestNeighborRecall(size_t d, size_t n_c, size_t per, uint32_t seed) {
    SeparatedSet s = MakeSeparated(d, n_c, per, seed);
    RaBitQQuantizer q;
    q.Fit(s.queries.data(), s.n_q, d);
    const size_t cs = q.CodeSize(d);
    std::vector<uint8_t> codes(s.n_q * cs);
    q.Encode(s.queries.data(), codes.data(), s.n_q, d);

    std::vector<uint32_t> knn(s.n_q);
    std::vector<float> dist(s.n_q);
    q.FindNearestNeighbor(
        codes.data(), nullptr, nullptr, s.centroids.data(),
        s.n_q, s.n_c, d, nullptr, nullptr, knn.data(), dist.data(), nullptr
    );
    size_t hits = 0;
    for (size_t i = 0; i < s.n_q; ++i) if (knn[i] == s.true_nn[i]) hits++;
    return static_cast<double>(hits) / static_cast<double>(s.n_q);
}

} // namespace

// ── RaBitQQuantizer unit tests ──

class RaBitQQuantizerTest : public ::testing::Test {
  protected:
    static constexpr size_t n = 2000;
    static constexpr size_t d = 64;
    std::vector<float> data;

    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n * d);
        for (auto& v : data) v = dist(rng);
    }
};

TEST_F(RaBitQQuantizerTest, InvalidDimensionality_Throws) {
    std::vector<float> buf(8192, 0.1f);
    // Not a multiple of 8.
    EXPECT_THROW(RaBitQQuantizer().Fit(buf.data(), 4, 60), std::invalid_argument);
    // Multiple of 8 but above RABITQ_MAX_DIMS (4096).
    EXPECT_THROW(RaBitQQuantizer().Fit(buf.data(), 1, RABITQ_MAX_DIMS + 8), std::invalid_argument);
    // Valid.
    EXPECT_NO_THROW(RaBitQQuantizer().Fit(buf.data(), 4, 64));
}

TEST_F(RaBitQQuantizerTest, CodeSize_Matches) {
    RaBitQQuantizer q;
    q.Fit(data.data(), n, d);
    EXPECT_EQ(q.CodeSize(d), (d + 7) / 8 + 8);
}

TEST_F(RaBitQQuantizerTest, ComputeNorms_MatchDistanceToCentroid) {
    RaBitQQuantizer q;
    q.Fit(data.data(), n, d);
    const size_t cs = q.CodeSize(d);
    std::vector<uint8_t> codes(n * cs);
    q.Encode(data.data(), codes.data(), n, d);

    std::vector<float> norms(n);
    q.ComputeNorms(codes.data(), n, d, norms.data());

    std::vector<float> mean = ColumnMean(data, n, d);
    for (size_t i = 0; i < std::min(n, size_t{200}); ++i) {
        double expected = 0.0;
        for (size_t j = 0; j < d; ++j) {
            double diff = static_cast<double>(data[i * d + j]) - mean[j];
            expected += diff * diff;
        }
        EXPECT_NEAR(norms[i], static_cast<float>(expected),
                    static_cast<float>(expected) * 0.05f + 1e-1f)
            << "vector " << i;
    }
}

TEST_F(RaBitQQuantizerTest, Decode_FinitePreservesDirection) {
    RaBitQQuantizer q;
    q.Fit(data.data(), n, d);
    const size_t cs = q.CodeSize(d);
    std::vector<uint8_t> codes(n * cs);
    q.Encode(data.data(), codes.data(), n, d);
    std::vector<float> decoded(n * d);
    q.Decode(codes.data(), decoded.data(), n, d);

    std::vector<float> mean = ColumnMean(data, n, d);
    size_t positive = 0;
    for (size_t i = 0; i < n; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < d; ++j) {
            float v = decoded[i * d + j];
            ASSERT_TRUE(std::isfinite(v)) << "vector " << i << " dim " << j;
            dot += (static_cast<double>(data[i * d + j]) - mean[j]) *
                   (static_cast<double>(v) - mean[j]);
        }
        if (dot > 0.0) positive++;
    }
    EXPECT_GT(static_cast<double>(positive) / n, 0.9);
}

TEST_F(RaBitQQuantizerTest, FindNearestNeighbor_MatchesBruteForce) {
    EXPECT_GT(NearestNeighborRecall(64, 20, 30, 7), 0.9);
}

TEST_F(RaBitQQuantizerTest, FindNearestNeighbor_HighDimensionNoOverflow) {
    // d within the RABITQ_MAX_DIMS cap must still produce correct results
    // (guards the uint16 FastScan accumulator boundary).
    EXPECT_GT(NearestNeighborRecall(2048, 16, 20, 11), 0.9);
    EXPECT_GT(NearestNeighborRecall(RABITQ_MAX_DIMS, 16, 12, 13), 0.9);
}

// ── SuperKMeans<u8> integration tests (RaBitQ) ──

class SuperKMeansRaBitQTest : public ::testing::Test {};

TEST_F(SuperKMeansRaBitQTest, BasicTraining) {
    const size_t n = 2000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::rabitq;

    auto kmeans = skm_u8(n_clusters, d, config);
    EXPECT_FALSE(kmeans.IsTrained());
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), n_clusters * d);
    EXPECT_EQ(kmeans.GetNClusters(), n_clusters);
}

TEST_F(SuperKMeansRaBitQTest, AllClustersUsed_AssignmentsValid) {
    const size_t n = 5000, d = 128, n_clusters = 20;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 15;
    config.quantizer_type = QuantizerType::rabitq;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    EXPECT_EQ(assignments.size(), n);
    for (size_t i = 0; i < n; ++i) EXPECT_LT(assignments[i], n_clusters) << "at " << i;

    std::unordered_set<uint32_t> used(assignments.begin(), assignments.end());
    EXPECT_EQ(used.size(), n_clusters);
}

TEST_F(SuperKMeansRaBitQTest, Recall_MatchesGroundTruth) {
    float recall = skm_test::ClusteringRecall<Quantization::u8>(
        QuantizerType::rabitq, CMAKE_SOURCE_DIR "/tests/test_data.bin"
    );
    EXPECT_NEAR(recall, skm_test::RECALL_GROUND_TRUTH.at("rabitq"), skm_test::RECALL_TOL);
}

TEST_F(SuperKMeansRaBitQTest, QuantizedCentroidUpdate_Converges) {
    const size_t n = 3000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::rabitq;
    config.quantized_centroid_update = true;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_EQ(centroids.size(), n_clusters * d);

    auto stats = kmeans.GetIterationStats();
    ASSERT_GE(stats.size(), 2u);
    EXPECT_LT(stats.back().objective, stats.front().objective);
}

TEST_F(SuperKMeansRaBitQTest, QuantizedAssign_MatchesAssign) {
    const size_t n = 3000, d = 64, n_clusters = 12;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, false, 1.0f, 20.0f);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::rabitq;

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

TEST_F(SuperKMeansRaBitQTest, QuantizedAssign_RepeatedDifferentData) {
    const size_t n = 3000, d = 64, n_clusters = 12;
    std::vector<float> dataA = MakeBlobs(n, d, n_clusters, false, 1.0f, 20.0f, 1);
    std::vector<float> dataB = MakeBlobs(n, d, n_clusters, false, 1.0f, 20.0f, 999);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::rabitq;

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

TEST_F(SuperKMeansRaBitQTest, InvalidInputs_Throw) {
    const size_t n = 10000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);
    auto make = [&](const SuperKMeansConfig& c, size_t k, size_t dd) { return skm_u8(k, dd, c); };
    SuperKMeansConfig base;
    base.quantizer_type = QuantizerType::rabitq;

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

TEST_F(SuperKMeansRaBitQTest, EarlyTermination) {
    const size_t n = 10000, d = 64, n_clusters = 5, max_iters = 100;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, false, 0.5f, 20.0f);

    SuperKMeansConfig c_early;
    c_early.iters = max_iters;
    c_early.early_termination = true;
    c_early.tol = 1e-2f;
    c_early.sampling_fraction = 1.0f;
    c_early.quantizer_type = QuantizerType::rabitq;
    auto km_early = skm_u8(n_clusters, d, c_early);
    km_early.Train(data.data(), n);
    size_t iters_early = km_early.iteration_stats.size();

    SuperKMeansConfig c_no;
    c_no.iters = max_iters;
    c_no.early_termination = false;
    c_no.sampling_fraction = 1.0f;
    c_no.quantizer_type = QuantizerType::rabitq;
    auto km_no = skm_u8(n_clusters, d, c_no);
    km_no.Train(data.data(), n);
    size_t iters_no = km_no.iteration_stats.size();

    EXPECT_LT(iters_early, max_iters);
    EXPECT_EQ(iters_no, max_iters);
    EXPECT_LT(iters_early, iters_no);
}

TEST_F(SuperKMeansRaBitQTest, AngularMode_NormalizesCentroids) {
    const size_t n = 5000, d = 64, n_clusters = 50;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, true);

    SuperKMeansConfig config;
    config.iters = 10;
    config.angular = true;
    config.quantizer_type = QuantizerType::rabitq;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    for (size_t c = 0; c < n_clusters; ++c) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) norm += centroids[c * d + j] * centroids[c * d + j];
        EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-4f) << "centroid " << c;
    }
}

TEST_F(SuperKMeansRaBitQTest, Determinism_SameSeedSameCentroids) {
    const size_t n = 3000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.early_termination = false;
    config.seed = 123;
    config.n_threads = 1;
    config.quantizer_type = QuantizerType::rabitq;

    auto c1 = skm_u8(n_clusters, d, config).Train(data.data(), n);
    auto c2 = skm_u8(n_clusters, d, config).Train(data.data(), n);
    ASSERT_EQ(c1.size(), c2.size());
    for (size_t i = 0; i < c1.size(); ++i) EXPECT_FLOAT_EQ(c1[i], c2[i]) << "at " << i;
}

// ── Pruning integration tests (need d >= 128 and k > 256) ──

class SuperKMeansRaBitQPruningTest : public ::testing::Test {};

TEST_F(SuperKMeansRaBitQPruningTest, SLOW_PruningConverges) {
    const size_t n = 10000, d = 128, n_clusters = 300;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 5;
    config.quantizer_type = QuantizerType::rabitq;
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

TEST_F(SuperKMeansRaBitQPruningTest, SLOW_PruningRecallCloseToNoPruning) {
    const std::string path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    float pruned = skm_test::ClusteringRecall<Quantization::u8>(QuantizerType::rabitq, path, 300, 1024, true);
    float unpruned = skm_test::ClusteringRecall<Quantization::u8>(QuantizerType::rabitq, path, 300, 1024, false);
    EXPECT_NEAR(pruned, unpruned, skm_test::RECALL_PRUNE_TOL)
        << "pruned=" << pruned << " unpruned=" << unpruned;
}

TEST_F(SuperKMeansRaBitQPruningTest, SLOW_AssignTrainingPointsReuseMatchesQuantizedAssign) {
    const size_t n = 5000, d = 128, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::rabitq;
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
