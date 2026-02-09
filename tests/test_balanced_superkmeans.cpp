#include <gtest/gtest.h>
#include <omp.h>

#include "superkmeans/balanced_superkmeans.h"

class BalancedSuperKMeansTest : public ::testing::Test {
  protected:
    void SetUp() override { omp_set_num_threads(omp_get_max_threads()); }
};

TEST_F(BalancedSuperKMeansTest, ConfigSynchronizationWithParent) {
    const size_t n_clusters = 256;
    const size_t d = 128;

    skmeans::BalancedSuperKMeansConfig config;
    config.data_already_rotated = true;
    config.unrotate_centroids = true;
    config.iters_mesoclustering = 7;
    config.iters_fineclustering = 9;
    config.iters_refinement = 3;
    config.seed = 123;
    config.sampling_fraction = 0.5f;

    auto kmeans = skmeans::BalancedSuperKMeans<
        skmeans::Quantization::f32,
        skmeans::DistanceFunction::l2
    >(n_clusters, d, config);

    EXPECT_FALSE(kmeans.balanced_config.unrotate_centroids)
        << "unrotate_centroids should be forced to false when data_already_rotated=true";

    EXPECT_EQ(kmeans.balanced_config.iters_mesoclustering, 7)
        << "iters_mesoclustering should not be modified";
    EXPECT_EQ(kmeans.balanced_config.iters_fineclustering, 9)
        << "iters_fineclustering should not be modified";
    EXPECT_EQ(kmeans.balanced_config.iters_refinement, 3)
        << "iters_refinement should not be modified";

    EXPECT_EQ(kmeans.balanced_config.seed, 123)
        << "Other base config fields should be preserved";
    EXPECT_EQ(kmeans.balanced_config.sampling_fraction, 0.5f)
        << "Other base config fields should be preserved";
}
