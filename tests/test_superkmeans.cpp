#include <gtest/gtest.h>
#include <omp.h>
#include <random>
#include <unordered_set>
#include <vector>

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

class SuperKMeansTest : public ::testing::Test {
  protected:
    void SetUp() override {}
};

TEST_F(SuperKMeansTest, BasicTraining_SmallDataset) {
    const size_t n = 1000;
    const size_t d = 32;
    const size_t n_clusters = 10;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

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

TEST_F(SuperKMeansTest, AllClustersUsed) {
    const size_t n = 10000;
    const size_t d = 128;
    const size_t n_clusters = 50;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

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

TEST_F(SuperKMeansTest, PerformAssignments_PopulatesAssignments) {
    const size_t n = 1000;
    const size_t d = 32;
    const size_t n_clusters = 10;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

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

TEST_F(SuperKMeansTest, InvalidInputs_ThrowExceptions) {
    const size_t n = 10000;
    const size_t d = 32;
    const size_t n_clusters = 10;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    // More clusters than data points
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

    // Not enough samples to train (sampling_fraction too low)
    EXPECT_THROW(
        ([&]() {
            skmeans::SuperKMeansConfig config;
            config.sampling_fraction = 0.0001f; // Very low sampling fraction
            config.max_points_per_cluster = 1;  // Low max points per cluster
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, config
                );
            kmeans.Train(data.data(), n);
        }()),
        std::runtime_error
    );

    // Zero n_clusters
    EXPECT_THROW(
        ([&]() {
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    0, d
                );
        }()),
        std::invalid_argument
    );

    // Zero dimensionality
    EXPECT_THROW(
        ([&]() {
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, 0
                );
        }()),
        std::invalid_argument
    );

    // Zero iterations in config
    EXPECT_THROW(
        ([&]() {
            skmeans::SuperKMeansConfig config;
            config.iters = 0;
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, config
                );
        }()),
        std::invalid_argument
    );

    // Zero sampling_fraction
    EXPECT_THROW(
        ([&]() {
            skmeans::SuperKMeansConfig config;
            config.sampling_fraction = 0.0f;
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, config
                );
        }()),
        std::invalid_argument
    );

    // Negative sampling_fraction
    EXPECT_THROW(
        ([&]() {
            skmeans::SuperKMeansConfig config;
            config.sampling_fraction = -0.5f;
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, config
                );
        }()),
        std::invalid_argument
    );

    // sampling_fraction > 1.0
    EXPECT_THROW(
        ([&]() {
            skmeans::SuperKMeansConfig config;
            config.sampling_fraction = 1.5f;
            auto kmeans =
                skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                    n_clusters, d, config
                );
        }()),
        std::invalid_argument
    );

    // Training twice
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

/**
 * @brief Early termination stops when centroid shift falls below tolerance
 *
 */
TEST_F(SuperKMeansTest, EarlyTermination_ShiftBelowTol_Stops) {
    const size_t n = 10000;
    const size_t d = 64;
    const size_t n_clusters = 5;
    const size_t max_iters = 100;

    // Generate well-separated blobs with low variance for faster convergence
    std::mt19937 gen(42);
    std::vector<float> data(n * d);
    std::vector<std::vector<float>> centers(n_clusters, std::vector<float>(d));
    for (size_t c = 0; c < n_clusters; ++c) {
        for (size_t j = 0; j < d; ++j) {
            centers[c][j] = static_cast<float>(c) * 20.0f + (j % 2 == 0 ? 5.0f : -5.0f);
        }
    }
    std::normal_distribution<float> noise(0.0f, 0.5f);
    for (size_t i = 0; i < n; ++i) {
        size_t cluster = i % n_clusters;
        for (size_t j = 0; j < d; ++j) {
            data[i * d + j] = centers[cluster][j] + noise(gen);
        }
    }

    // Test WITH early termination
    skmeans::SuperKMeansConfig config_early;
    config_early.iters = max_iters;
    config_early.early_termination = true;
    config_early.tol = 1e-2f;
    config_early.verbose = false;
    config_early.seed = 42;
    config_early.sampling_fraction = 1.0f;
    auto kmeans_early =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, config_early
        );
    kmeans_early.Train(data.data(), n);
    const auto& stats_early = kmeans_early.iteration_stats;
    size_t iters_with_early = stats_early.size();

    // Test WITHOUT early termination
    skmeans::SuperKMeansConfig config_no_early;
    config_no_early.iters = max_iters;
    config_no_early.early_termination = false;
    config_no_early.verbose = false;
    config_no_early.seed = 42;
    config_no_early.sampling_fraction = 1.0f;
    auto kmeans_no_early =
        skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
            n_clusters, d, config_no_early
        );
    kmeans_no_early.Train(data.data(), n);
    const auto& stats_no_early = kmeans_no_early.iteration_stats;
    size_t iters_without_early = stats_no_early.size();

    // With early termination, should stop before max_iters
    EXPECT_LT(iters_with_early, max_iters)
        << "Early termination should stop before max_iters=" << max_iters;

    // Without early termination, should run all iterations
    EXPECT_EQ(iters_without_early, max_iters)
        << "Without early termination, should run all " << max_iters << " iterations";

    // Early termination should use fewer iterations
    EXPECT_LT(iters_with_early, iters_without_early)
        << "Early termination (" << iters_with_early << " iters) should use fewer iterations "
        << "than no early termination (" << iters_without_early << " iters)";
}

/**
 * @brief Test that disabling early termination runs all iterations
 */
TEST_F(SuperKMeansTest, EarlyTermination_Disabled_RunsAllIterations) {
    const size_t n = 10000;
    const size_t d = 32;
    const size_t n_clusters = 5;
    const size_t max_iters = 50;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::SuperKMeansConfig config;
    config.iters = max_iters;
    config.early_termination = false;
    config.verbose = false;
    config.sampling_fraction = 1.0f;

    auto kmeans = skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
        n_clusters, d, config
    );
    kmeans.Train(data.data(), n);

    const auto& stats = kmeans.iteration_stats;
    EXPECT_EQ(stats.size(), max_iters)
        << "With early_termination=false, should run exactly " << max_iters << " iterations";
}

/**
 * @brief Test: Sampling provides significant speedup over full data training
 *
 * Verifies that using sampling_fraction=0.1 is at least 2x faster than
 * training on the full dataset (sampling_fraction=1.0).
 */
TEST_F(SuperKMeansTest, Sampling_ProvidesSpeedup) {
    const size_t n = 50000;
    const size_t d = 128;
    const size_t n_clusters = 500;
    const size_t n_runs = 10;

    std::vector<float> data = skmeans::MakeBlobs(n, d, n_clusters);

    skmeans::SuperKMeansConfig base_config;
    base_config.iters = 25;
    base_config.early_termination = false;
    base_config.angular = false;
    base_config.verbose = false;

    skmeans::TicToc timer_full;
    for (size_t i = 0; i < n_runs; ++i) {
        skmeans::SuperKMeansConfig config = base_config;
        config.sampling_fraction = 1.0f;
        config.seed = static_cast<uint32_t>(42 + i);

        auto kmeans =
            skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                n_clusters, d, config
            );

        timer_full.Tic();
        kmeans.Train(data.data(), n);
        timer_full.Toc();
    }

    // 10% sampling
    skmeans::TicToc timer_sampled;
    for (size_t i = 0; i < n_runs; ++i) {
        skmeans::SuperKMeansConfig config = base_config;
        config.sampling_fraction = 0.1f;
        config.seed = static_cast<uint32_t>(42 + i);

        auto kmeans =
            skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>(
                n_clusters, d, config
            );

        timer_sampled.Tic();
        kmeans.Train(data.data(), n);
        timer_sampled.Toc();
    }

    double full_time_ms = timer_full.accum_time / 1e6;
    double sampled_time_ms = timer_sampled.accum_time / 1e6;
    double speedup = full_time_ms / sampled_time_ms;

    // Sampling should be at least 2x faster
    EXPECT_GE(speedup, 2.0)
        << "Sampling should provide at least 2x speedup. "
        << "Full: " << full_time_ms << "ms, Sampled: " << sampled_time_ms << "ms, "
        << "Speedup: " << speedup << "x";
}
