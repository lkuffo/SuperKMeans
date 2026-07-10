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
#include "superkmeans/quantizers/sq8.h"
#include "superkmeans/superkmeans.h"

#include "recall_utils.h"

using namespace skmeans;

namespace {

using skm_u8 = SuperKMeans<Quantization::u8, DistanceFunction::l2>;

} // namespace

// ── SQ8Quantizer unit tests ──

class SQ8QuantizerTest : public ::testing::Test {
  protected:
    static constexpr size_t n = 1000;
    static constexpr size_t d = 128;
    std::vector<float> data;

    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        data.resize(n * d);
        for (auto& v : data) v = dist(rng);
    }
};

TEST_F(SQ8QuantizerTest, FitEncodeDecode_Roundtrip) {
    SQ8Quantizer quantizer;
    EXPECT_FALSE(quantizer.IsFitted());
    quantizer.Fit(data.data(), n, d);
    EXPECT_TRUE(quantizer.IsFitted());
    EXPECT_EQ(quantizer.CodeSize(d), d);

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);
    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded.data(), decoded.data(), n, d);

    const auto& params = quantizer.GetParams();
    float max_err = params.inv_quantization_scale;
    for (size_t i = 0; i < n * d; ++i) {
        EXPECT_NEAR(data[i], decoded[i], max_err + 1e-5f) << "at index " << i;
    }
}

TEST_F(SQ8QuantizerTest, ComputeNorms_ConsistentWithDistances) {
    SQ8Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);

    std::vector<uint8_t> encoded(n * d);
    quantizer.Encode(data.data(), encoded.data(), n, d);
    std::vector<float> q_norms(n);
    quantizer.ComputeNorms(encoded.data(), n, d, q_norms.data());

    const auto& params = quantizer.GetParams();
    float inv_scale_sq = params.inv_quantization_scale * params.inv_quantization_scale;
    for (size_t i = 0; i < std::min(n, size_t{100}); ++i) {
        uint32_t sum_sq = 0;
        for (size_t j = 0; j < d; ++j) {
            uint32_t v = encoded[i * d + j];
            sum_sq += v * v;
        }
        float expected_norm = inv_scale_sq * static_cast<float>(sum_sq);
        EXPECT_FLOAT_EQ(q_norms[i], expected_norm) << "norm mismatch at vector " << i;
    }
}

TEST_F(SQ8QuantizerTest, FindNearestNeighbor_MatchesBruteForce) {
    SQ8Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);

    size_t n_centroids = 50;
    size_t n_queries = 200;

    std::vector<uint8_t> encoded_data(n * d);
    quantizer.Encode(data.data(), encoded_data.data(), n, d);
    std::vector<float> q_norms(n);
    quantizer.ComputeNorms(encoded_data.data(), n, d, q_norms.data());

    const uint8_t* queries = encoded_data.data() + n_centroids * d;
    const uint8_t* centroids = encoded_data.data();
    const float* query_norms = q_norms.data() + n_centroids;
    const float* centroid_norms = q_norms.data();

    std::vector<uint32_t> knn(n_queries);
    std::vector<float> distances(n_queries);
    std::vector<float> tmp_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

    quantizer.FindNearestNeighbor(
        queries, centroids, data.data() + n_centroids * d, data.data(),
        n_queries, n_centroids, d,
        query_norms, centroid_norms, knn.data(), distances.data(), tmp_buf.data()
    );

    std::vector<float> decoded(n * d);
    quantizer.Decode(encoded_data.data(), decoded.data(), n, d);
    for (size_t i = 0; i < n_queries; ++i) {
        float best_dist = std::numeric_limits<float>::max();
        uint32_t best_idx = 0;
        for (size_t j = 0; j < n_centroids; ++j) {
            float dist = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                float diff = decoded[(n_centroids + i) * d + k] - decoded[j * d + k];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_idx = static_cast<uint32_t>(j); }
        }
        EXPECT_EQ(knn[i], best_idx) << "query " << i;
    }
}

TEST_F(SQ8QuantizerTest, Encode_ClampsOutOfRange) {
    SQ8Quantizer quantizer;
    quantizer.Fit(data.data(), n, d);

    std::vector<float> extreme(d);
    for (size_t j = 0; j < d; ++j) extreme[j] = (j % 2 == 0) ? -1e6f : 1e6f;
    std::vector<uint8_t> codes(d);
    quantizer.Encode(extreme.data(), codes.data(), 1, d);
    for (size_t j = 0; j < d; ++j) {
        EXPECT_EQ(codes[j], (j % 2 == 0) ? 0u : 255u) << "no clamp at dim " << j;
    }
}

// ── FinalizeCentroids unit test ──

TEST(SQ8FinalizeCentroidsTest, RecoversClusterMeans) {
    SQ8Quantizer quantizer;
    const size_t d = 8;
    const size_t n_clusters = 3;
    const size_t n_vectors = 6;

    std::vector<float> fit_data(100 * d);
    for (size_t i = 0; i < fit_data.size(); ++i) fit_data[i] = static_cast<float>(i % 256) / 255.0f;
    quantizer.Fit(fit_data.data(), 100, d);

    std::vector<float> vectors(n_vectors * d);
    for (size_t j = 0; j < 2 * d; ++j) vectors[j] = 0.5f;
    for (size_t j = 2 * d; j < 5 * d; ++j) vectors[j] = 0.25f;
    for (size_t j = 5 * d; j < 6 * d; ++j) vectors[j] = 0.75f;

    std::vector<uint8_t> encoded(n_vectors * d);
    quantizer.Encode(vectors.data(), encoded.data(), n_vectors, d);

    std::vector<uint32_t> assignments = {0, 0, 1, 1, 1, 2};
    std::vector<float> centroid_buf(n_clusters * d, 0.0f);
    std::vector<uint32_t> cluster_sizes(n_clusters, 0);

    quantizer.ResetCentroidAccumulators(n_clusters, d);
    quantizer.UpdateCentroids(
        encoded.data(), assignments.data(), centroid_buf.data(), cluster_sizes.data(),
        n_vectors, n_clusters, d, 1
    );
    quantizer.FinalizeCentroids(centroid_buf.data(), cluster_sizes.data(), n_clusters, d);

    EXPECT_EQ(cluster_sizes[0], 2u);
    EXPECT_EQ(cluster_sizes[1], 3u);
    EXPECT_EQ(cluster_sizes[2], 1u);
    for (size_t j = 0; j < d; ++j) {
        EXPECT_NEAR(centroid_buf[0 * d + j], 0.5f, 0.02f) << "cluster 0, dim " << j;
        EXPECT_NEAR(centroid_buf[1 * d + j], 0.25f, 0.02f) << "cluster 1, dim " << j;
        EXPECT_NEAR(centroid_buf[2 * d + j], 0.75f, 0.02f) << "cluster 2, dim " << j;
    }
}

// ── SuperKMeans<u8> integration tests ──

class SuperKMeansSQ8Test : public ::testing::Test {};

TEST_F(SuperKMeansSQ8Test, BasicTraining) {
    const size_t n = 2000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::sq8;

    auto kmeans = skm_u8(n_clusters, d, config);
    EXPECT_FALSE(kmeans.IsTrained());
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), n_clusters * d);
    EXPECT_EQ(kmeans.GetNClusters(), n_clusters);
}

TEST_F(SuperKMeansSQ8Test, AllClustersUsed_AssignmentsValid) {
    const size_t n = 5000, d = 128, n_clusters = 20;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 15;
    config.quantizer_type = QuantizerType::sq8;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    auto assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    EXPECT_EQ(assignments.size(), n);
    for (size_t i = 0; i < n; ++i) EXPECT_LT(assignments[i], n_clusters) << "at " << i;

    std::unordered_set<uint32_t> used(assignments.begin(), assignments.end());
    EXPECT_EQ(used.size(), n_clusters);
}

TEST_F(SuperKMeansSQ8Test, Recall_MatchesGroundTruth) {
    float recall = skm_test::ClusteringRecall<Quantization::u8>(
        QuantizerType::sq8, CMAKE_SOURCE_DIR "/tests/test_data.bin"
    );
    EXPECT_NEAR(recall, skm_test::RECALL_GROUND_TRUTH.at("sq8"), skm_test::RECALL_TOL);
}

TEST_F(SuperKMeansSQ8Test, QuantizedCentroidUpdate_Converges) {
    const size_t n = 3000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::sq8;
    config.quantized_centroid_update = true;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    EXPECT_EQ(centroids.size(), n_clusters * d);

    auto stats = kmeans.GetIterationStats();
    ASSERT_GE(stats.size(), 2u);
    EXPECT_LT(stats.back().objective, stats.front().objective);
}

TEST_F(SuperKMeansSQ8Test, QuantizedAssign_MatchesAssign_Balanced) {
    const size_t n = 3000, d = 128, n_clusters = 15;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 5;
    config.quantizer_type = QuantizerType::sq8;
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
    for (size_t i = 0; i < n; ++i) if (assign_gt[i] == assign_q[i]) agree++;
    EXPECT_GT(static_cast<double>(agree) / n, 0.5);
}

TEST_F(SuperKMeansSQ8Test, QuantizedAssign_RepeatedDifferentData) {
    const size_t n = 3000, d = 64, n_clusters = 12;
    std::vector<float> dataA = MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 1);
    std::vector<float> dataB = MakeBlobs(n, d, n_clusters, false, 1.0f, 10.0f, 999);

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::sq8;

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

TEST_F(SuperKMeansSQ8Test, InvalidInputs_Throw) {
    const size_t n = 10000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);
    auto make = [&](const SuperKMeansConfig& c, size_t k, size_t dd) { return skm_u8(k, dd, c); };
    SuperKMeansConfig base;
    base.quantizer_type = QuantizerType::sq8;

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

TEST_F(SuperKMeansSQ8Test, EarlyTermination) {
    const size_t n = 10000, d = 64, n_clusters = 5, max_iters = 100;
    std::vector<float> data = MakeBlobs(n, d, n_clusters, false, 0.5f, 20.0f);

    SuperKMeansConfig c_early;
    c_early.iters = max_iters;
    c_early.early_termination = true;
    c_early.tol = 1e-2f;
    c_early.sampling_fraction = 1.0f;
    c_early.quantizer_type = QuantizerType::sq8;
    auto km_early = skm_u8(n_clusters, d, c_early);
    km_early.Train(data.data(), n);
    size_t iters_early = km_early.iteration_stats.size();

    SuperKMeansConfig c_no;
    c_no.iters = max_iters;
    c_no.early_termination = false;
    c_no.sampling_fraction = 1.0f;
    c_no.quantizer_type = QuantizerType::sq8;
    auto km_no = skm_u8(n_clusters, d, c_no);
    km_no.Train(data.data(), n);
    size_t iters_no = km_no.iteration_stats.size();

    EXPECT_LT(iters_early, max_iters);
    EXPECT_EQ(iters_no, max_iters);
    EXPECT_LT(iters_early, iters_no);
}

TEST_F(SuperKMeansSQ8Test, AngularMode_NormalizesCentroids) {
    const size_t n = 5000, d = 64, n_clusters = 50;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.angular = true;
    config.quantizer_type = QuantizerType::sq8;

    auto kmeans = skm_u8(n_clusters, d, config);
    auto centroids = kmeans.Train(data.data(), n);
    for (size_t c = 0; c < n_clusters; ++c) {
        float norm = 0.0f;
        for (size_t j = 0; j < d; ++j) norm += centroids[c * d + j] * centroids[c * d + j];
        EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-4f) << "centroid " << c;
    }
}

TEST_F(SuperKMeansSQ8Test, Determinism_SameSeedSameCentroids) {
    const size_t n = 3000, d = 64, n_clusters = 10;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 10;
    config.early_termination = false;
    config.seed = 123;
    config.quantizer_type = QuantizerType::sq8;

    auto c1 = skm_u8(n_clusters, d, config).Train(data.data(), n);
    auto c2 = skm_u8(n_clusters, d, config).Train(data.data(), n);
    ASSERT_EQ(c1.size(), c2.size());
    for (size_t i = 0; i < c1.size(); ++i) EXPECT_FLOAT_EQ(c1[i], c2[i]) << "at " << i;
}

// ── Pruning integration tests (need d >= 128 and k > 256) ──

class SuperKMeansSQ8PruningTest : public ::testing::Test {};

TEST_F(SuperKMeansSQ8PruningTest, SLOW_PruningConverges) {
    const size_t n = 10000, d = 128, n_clusters = 300;
    std::vector<float> data = MakeBlobs(n, d, n_clusters);

    SuperKMeansConfig config;
    config.iters = 5;
    config.quantizer_type = QuantizerType::sq8;
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

TEST_F(SuperKMeansSQ8PruningTest, SLOW_PruningRecallCloseToNoPruning) {
    const std::string path = CMAKE_SOURCE_DIR "/tests/test_data.bin";
    float pruned = skm_test::ClusteringRecall<Quantization::u8>(QuantizerType::sq8, path, 300, 1024, true);
    float unpruned = skm_test::ClusteringRecall<Quantization::u8>(QuantizerType::sq8, path, 300, 1024, false);
    EXPECT_NEAR(pruned, unpruned, skm_test::RECALL_PRUNE_TOL)
        << "pruned=" << pruned << " unpruned=" << unpruned;
}

TEST_F(SuperKMeansSQ8PruningTest, SLOW_AssignTrainingPointsReuseMatchesQuantizedAssign) {
    const size_t n = 5000, d = 128, n_clusters = 300;
    auto data = skm_test::LoadTestDataSubdim(
        CMAKE_SOURCE_DIR "/tests/test_data.bin", n, skm_test::RECALL_D, d
    );

    SuperKMeansConfig config;
    config.iters = 10;
    config.quantizer_type = QuantizerType::sq8;
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
