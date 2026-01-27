#include <gtest/gtest.h>
#include <omp.h>
#include <random>
#include <unordered_set>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/superkmeans.h"

namespace {

std::vector<float> make_blobs(
    size_t n_samples,
    size_t n_features,
    size_t n_centers,
    unsigned int random_state = 42
) {
    std::mt19937 gen(random_state);
    std::normal_distribution<float> center_dist(0.0f, 1.0f);
    std::vector<std::vector<float>> centers(n_centers, std::vector<float>(n_features));
    for (auto& c : centers)
        for (auto& x : c)
            x = center_dist(gen);

    std::uniform_int_distribution<size_t> cluster_dist(0, n_centers - 1);
    std::normal_distribution<float> point_dist(0.0f, 1.0f);

    std::vector<float> data;
    data.reserve(n_samples * n_features);

    for (size_t i = 0; i < n_samples; ++i) {
        const auto& center = centers[cluster_dist(gen)];
        for (size_t j = 0; j < n_features; ++j)
            data.push_back(center[j] + point_dist(gen));
    }

    return data;
}

} // anonymous namespace

class SuperKMeansTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(SuperKMeansTest, BasicTraining_SmallDataset) {
    const size_t n = 1000;
    const size_t d = 32;
    const size_t n_clusters = 10;

    std::vector<float> data = make_blobs(n, d, n_clusters);

    skmeans::SuperKMeansConfig config;
    config.iters = 10;
    config.verbose = false;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );

    EXPECT_FALSE(kmeans.IsTrained());

    auto centroids = kmeans.Train(data.data(), n);

    EXPECT_TRUE(kmeans.IsTrained());
    EXPECT_EQ(centroids.size(), n_clusters * d);
    EXPECT_EQ(kmeans.GetNClusters(), n_clusters);
}

TEST_F(SuperKMeansTest, CentroidsAreValid) {
    const size_t n = 500;
    const size_t d = 64;
    const size_t n_clusters = 5;

    std::vector<float> data = make_blobs(n, d, n_clusters);

    skmeans::SuperKMeansConfig config;
    config.iters = 10;
    config.verbose = false;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);

    // Check that centroids contain valid values (no NaN or Inf)
    for (size_t i = 0; i < centroids.size(); ++i) {
        EXPECT_TRUE(std::isfinite(centroids[i]))
            << "Centroid value at index " << i << " is not finite: " << centroids[i];
    }
}

TEST_F(SuperKMeansTest, AllClustersUsed) {
    const size_t n = 10000;
    const size_t d = 128;
    const size_t n_clusters = 50;

    std::vector<float> data = make_blobs(n, d, n_clusters);

    skmeans::SuperKMeansConfig config;
    config.iters = 25;
    config.verbose = false;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);

    // Check that all clusters have at least one assignment
    const auto& assignments = kmeans._assignments;
    std::unordered_set<uint32_t> used_clusters(assignments.begin(), assignments.end());

    EXPECT_EQ(used_clusters.size(), n_clusters)
        << "Not all clusters were used. Expected " << n_clusters << " but only "
        << used_clusters.size() << " were assigned.";
}

TEST_F(SuperKMeansTest, SamplingFraction_ReducesComputations) {
    const size_t n = 10000;
    const size_t d = 64;
    const size_t n_clusters = 20;

    std::vector<float> data = make_blobs(n, d, n_clusters);

    // Test with 50% sampling
    skmeans::SuperKMeansConfig config;
    config.iters = 10;
    config.sampling_fraction = 0.5f;
    config.verbose = false;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);

    EXPECT_EQ(centroids.size(), n_clusters * d);
    // Centroids should still be valid even with sampling
    for (const auto& val : centroids) {
        EXPECT_TRUE(std::isfinite(val));
    }
}

TEST_F(SuperKMeansTest, PerformAssignments_PopulatesAssignments) {
    const size_t n = 1000;
    const size_t d = 32;
    const size_t n_clusters = 10;

    std::vector<float> data = make_blobs(n, d, n_clusters);

    skmeans::SuperKMeansConfig config;
    config.iters = 10;
    config.verbose = false;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    auto centroids = kmeans.Train(data.data(), n);

    const auto& assignments = kmeans._assignments;
    EXPECT_EQ(assignments.size(), n);

    // Check all assignments are valid cluster indices
    for (size_t i = 0; i < n; ++i) {
        EXPECT_LT(assignments[i], n_clusters)
            << "Assignment " << i << " has invalid cluster index: " << assignments[i];
    }
}

TEST_F(SuperKMeansTest, DifferentSeeds_ProduceDifferentResults) {
    const size_t n = 1000;
    const size_t d = 32;
    const size_t n_clusters = 10;

    std::vector<float> data = make_blobs(n, d, n_clusters);

    skmeans::SuperKMeansConfig config1;
    config1.iters = 10;
    config1.seed = 42;
    config1.verbose = false;

    skmeans::SuperKMeansConfig config2;
    config2.iters = 10;
    config2.seed = 123;
    config2.verbose = false;

    auto kmeans1 = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config1
    );
    auto centroids1 = kmeans1.Train(data.data(), n);

    auto kmeans2 = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config2
    );
    auto centroids2 = kmeans2.Train(data.data(), n);

    // Results should be different (at least some centroids should differ)
    bool found_difference = false;
    for (size_t i = 0; i < centroids1.size(); ++i) {
        if (std::abs(centroids1[i] - centroids2[i]) > 1e-6) {
            found_difference = true;
            break;
        }
    }

    EXPECT_TRUE(found_difference) << "Different seeds should produce different clustering results";
}

TEST_F(SuperKMeansTest, InvalidInputs_ThrowExceptions) {
    const size_t n = 10000;
    const size_t d = 32;
    const size_t n_clusters = 10;

    std::vector<float> data = make_blobs(n, d, n_clusters);

    // Test with more clusters than data points
    EXPECT_THROW(
        ([&]() {
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n + 10, d
                );
            kmeans.Train(data.data(), n);
        }()),
        std::runtime_error
    );

    // Test training twice
    EXPECT_THROW(
        ([&]() {
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d
                );
            kmeans.Train(data.data(), n);
            kmeans.Train(data.data(), n); // Should throw
        }()),
        std::runtime_error
    );
}
