#include <cmath>
#include <gtest/gtest.h>
#include <omp.h>
#include <unordered_set>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

namespace {

/**
 * @brief Compute squared L2 distance between two vectors
 */
float ComputeL2DistanceSquared(const float* a, const float* b, size_t d) {
    float dist = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 * @brief Find the index of the nearest centroid for a given point using brute force
 */
uint32_t FindNearestCentroidBruteForce(
    const float* point,
    const float* centroids,
    size_t n_clusters,
    size_t d
) {
    uint32_t best_idx = 0;
    float best_dist = std::numeric_limits<float>::max();

    for (size_t c = 0; c < n_clusters; ++c) {
        float dist = ComputeL2DistanceSquared(point, centroids + c * d, d);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<uint32_t>(c);
        }
    }
    return best_idx;
}

} // anonymous namespace

class AssignTest : public ::testing::Test {
  protected:
    void SetUp() override { omp_set_num_threads(omp_get_max_threads()); }
};

/**
 * @brief Test that Assign() produces same results as Train() assignments
 */
TEST_F(AssignTest, AssignMatchesTrainAssignments_SyntheticClusters) {
    const size_t n = 100000;
    const size_t d = 128;
    const size_t n_clusters = 1024;
    const int n_iters = 25;
    const float sampling_fraction = 1.0f;

    std::vector<float> data = skmeans::make_blobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = sampling_fraction;
    config.verbose = false;
    config.unrotate_centroids = true;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );

    auto centroids = kmeans.Train(data.data(), n);

    const auto& train_assignments = kmeans._assignments;
    auto assign_assignments = kmeans.Assign(data.data(), centroids.data(), n, n_clusters);

    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (train_assignments[i] != assign_assignments[i]) {
            ++mismatches;
        }
    }

    double mismatch_pct = 100.0 * static_cast<double>(mismatches) / static_cast<double>(n);
    EXPECT_LE(mismatch_pct, 0.01)
        << "Synthetic clusterable data should have very low mismatch rate, got " << mismatch_pct
        << "%";
}

/**
 * @brief Test that each point is assigned to its actual nearest centroid
 *
 * After training, we verify assignments by recomputing distances using brute force
 * and checking that each point is assigned to the centroid with minimum distance.
 */
TEST_F(AssignTest, EachPointAssignedToNearestCentroid) {
    const size_t n = 10000;  // Smaller n for brute-force verification
    const size_t d = 64;
    const size_t n_clusters = 100;
    const int n_iters = 10;

    std::vector<float> data = skmeans::make_blobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 42;
    config.unrotate_centroids = true;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );

    auto centroids = kmeans.Train(data.data(), n);
    const auto& assignments = kmeans._assignments;

    ASSERT_EQ(assignments.size(), n) << "Assignment size mismatch";
    ASSERT_EQ(centroids.size(), n_clusters * d) << "Centroid size mismatch";

    // Verify each assignment using brute-force nearest neighbor
    size_t incorrect_assignments = 0;
    for (size_t i = 0; i < n; ++i) {
        const float* point = data.data() + i * d;
        uint32_t assigned_cluster = assignments[i];
        uint32_t nearest_cluster = FindNearestCentroidBruteForce(point, centroids.data(), n_clusters, d);

        if (assigned_cluster != nearest_cluster) {
            // Double-check: maybe they have the same distance (tie-breaking)
            float assigned_dist = ComputeL2DistanceSquared(point, centroids.data() + assigned_cluster * d, d);
            float nearest_dist = ComputeL2DistanceSquared(point, centroids.data() + nearest_cluster * d, d);

            // Allow small floating point tolerance for ties
            if (std::abs(assigned_dist - nearest_dist) > 1e-4f * nearest_dist) {
                ++incorrect_assignments;
            }
        }
    }

    EXPECT_EQ(incorrect_assignments, 0)
        << "Found " << incorrect_assignments << " points not assigned to their nearest centroid";
}

/**
 * @brief Test that each point is assigned to its actual nearest centroid (high-dimensional)
 *
 * Same as above but with higher dimensions to test the DCT rotation path.
 */
TEST_F(AssignTest, EachPointAssignedToNearestCentroid_HighDim) {
    const size_t n = 5000;  // Smaller n for brute-force verification
    const size_t d = 512;   // High-dimensional (uses DCT rotation)
    const size_t n_clusters = 50;
    const int n_iters = 10;

    std::vector<float> data = skmeans::make_blobs(n, d, n_clusters, false, 1.0f, 10.0f, 123);

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 123;
    config.unrotate_centroids = true;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );

    auto centroids = kmeans.Train(data.data(), n);
    const auto& assignments = kmeans._assignments;

    ASSERT_EQ(assignments.size(), n) << "Assignment size mismatch";

    size_t incorrect_assignments = 0;
    for (size_t i = 0; i < n; ++i) {
        const float* point = data.data() + i * d;
        uint32_t assigned_cluster = assignments[i];
        uint32_t nearest_cluster = FindNearestCentroidBruteForce(point, centroids.data(), n_clusters, d);

        if (assigned_cluster != nearest_cluster) {
            float assigned_dist = ComputeL2DistanceSquared(point, centroids.data() + assigned_cluster * d, d);
            float nearest_dist = ComputeL2DistanceSquared(point, centroids.data() + nearest_cluster * d, d);

            if (std::abs(assigned_dist - nearest_dist) > 1e-4f * nearest_dist) {
                ++incorrect_assignments;
            }
        }
    }

    EXPECT_EQ(incorrect_assignments, 0)
        << "Found " << incorrect_assignments << " points not assigned to their nearest centroid (high-dim)";
}

/**
 * @brief Test that all clusters are non-empty after training
 *
 * K-means with proper empty cluster handling (split clusters) should ensure
 * that no cluster remains empty after training.
 */
TEST_F(AssignTest, AllClustersNonEmpty) {
    const size_t n = 50000;
    const size_t d = 128;
    const size_t n_clusters = 500;
    const int n_iters = 15;

    std::vector<float> data = skmeans::make_blobs(n, d, n_clusters, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 42;
    config.unrotate_centroids = true;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );

    auto centroids = kmeans.Train(data.data(), n);
    const auto& assignments = kmeans._assignments;

    ASSERT_EQ(assignments.size(), n) << "Assignment size mismatch";

    // Count points per cluster
    std::vector<size_t> cluster_counts(n_clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        ASSERT_LT(assignments[i], n_clusters) << "Invalid cluster index " << assignments[i];
        cluster_counts[assignments[i]]++;
    }

    // Check that all clusters have at least one point
    std::vector<size_t> empty_clusters;
    for (size_t c = 0; c < n_clusters; ++c) {
        if (cluster_counts[c] == 0) {
            empty_clusters.push_back(c);
        }
    }

    EXPECT_EQ(empty_clusters.size(), 0)
        << "Found " << empty_clusters.size() << " empty clusters out of " << n_clusters;

    // Also verify all clusters are used
    std::unordered_set<uint32_t> used_clusters(assignments.begin(), assignments.end());
    EXPECT_EQ(used_clusters.size(), n_clusters)
        << "Not all clusters were used. Expected " << n_clusters << " but only " << used_clusters.size()
        << " were assigned.";
}

/**
 * @brief Test that all clusters are non-empty with more clusters than data naturally supports
 *
 * When n_clusters > n_true_centers in make_blobs, some clusters would naturally be empty
 * without the split mechanism. This tests that split_clusters properly handles this.
 */
TEST_F(AssignTest, AllClustersNonEmpty_MoreClustersThanNaturalCenters) {
    const size_t n = 10000;
    const size_t d = 64;
    const size_t n_true_centers = 10;  // Only 10 natural clusters in the data
    const size_t n_clusters = 100;      // But we ask for 100 clusters
    const int n_iters = 15;

    std::vector<float> data = skmeans::make_blobs(n, d, n_true_centers, false, 1.0f, 10.0f, 42);

    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.sampling_fraction = 1.0f;
    config.verbose = false;
    config.seed = 42;
    config.unrotate_centroids = true;
    config.perform_assignments = true;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );

    auto centroids = kmeans.Train(data.data(), n);
    const auto& assignments = kmeans._assignments;

    // Count points per cluster
    std::vector<size_t> cluster_counts(n_clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        cluster_counts[assignments[i]]++;
    }

    // Check that all clusters have at least one point
    size_t empty_count = 0;
    for (size_t c = 0; c < n_clusters; ++c) {
        if (cluster_counts[c] == 0) {
            empty_count++;
        }
    }

    EXPECT_EQ(empty_count, 0)
        << "Found " << empty_count << " empty clusters when using " << n_clusters
        << " clusters on data with only " << n_true_centers << " natural centers";
}
