#pragma once

#include "superkmeans/superkmeans.h"

namespace skmeans {

/**
 * @brief Configuration parameters for Balanced SuperKMeans clustering.
 */
struct BalancedSuperKMeansConfig : SuperKMeansConfig {
    uint32_t n_points_per_mesocluster = 1024;
    uint32_t iters_mesoclustering = 10;
    float sampling_fraction_mesoclustering = 0.3f;
};

/**
 * @brief Statistics for BalancedSuperKMeans clustering.
 */
struct BalancedSuperKMeansIterationStats {
    std::vector<SuperKMeansIterationStats> mesoclustering_iteration_stats;
    std::vector<SuperKMeansIterationStats> refinement_iteration_stats;
    std::vector<std::vector<SuperKMeansIterationStats>> fineclustering_iteration_stats;
};

template <Quantization q = Quantization::f32, DistanceFunction alpha = DistanceFunction::l2>
class BalancedSuperKMeans : public SuperKMeans<q, alpha> {
    using typename SuperKMeans<q, alpha>::centroid_value_t;
    using typename SuperKMeans<q, alpha>::vector_value_t;
    using typename SuperKMeans<q, alpha>::distance_t;
    using typename SuperKMeans<q, alpha>::MatrixR;
    using typename SuperKMeans<q, alpha>::VectorR;
    using typename SuperKMeans<q, alpha>::pruner_t;
    using typename SuperKMeans<q, alpha>::layout_t;
    using typename SuperKMeans<q, alpha>::batch_computer;

  public:
    /**
     * @brief Constructor with custom configuration
     */
    BalancedSuperKMeans(
        size_t n_clusters,
        size_t dimensionality,
        const BalancedSuperKMeansConfig& config
    )
        : SuperKMeans<q, alpha>(n_clusters, dimensionality, config), balanced_config(config) {
        SKMEANS_ENSURE_POSITIVE(config.n_points_per_mesocluster);
        SKMEANS_ENSURE_POSITIVE(config.iters_mesoclustering);
        SKMEANS_ENSURE_POSITIVE(config.sampling_fraction_mesoclustering);

        if (n_clusters <= 128) {
            std::cout << "WARNING: n_clusters <= 128 is not recommended for BalancedSuperKMeans. "
                         "Consider using at least 128 clusters."
                      << std::endl;
        }
    }

    /**
     * @brief Default constructor
     */
    BalancedSuperKMeans(size_t n_clusters, size_t dimensionality)
        : BalancedSuperKMeans(n_clusters, dimensionality, BalancedSuperKMeansConfig{}) {}

    /**
     * @brief Run balanced k-means clustering to determine centroids
     *
     * @param data Pointer to the data matrix (row-major, n × d)
     * @param n Number of points (rows) in the data matrix
     * @param queries Optional pointer to query vectors for recall computation
     * @param n_queries Number of query vectors
     *
     * @return std::vector<skmeans_centroid_value_t<q>> Trained centroids
     */
    std::vector<skmeans_centroid_value_t<q>> Train(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        const vector_value_t* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0
    ) {
        // Presetup
        SKMEANS_ENSURE_POSITIVE(n);
        if (this->_trained) {
            throw std::runtime_error("The clustering has already been trained");
        }
        n_mesoclusters = static_cast<size_t>(std::floor(std::sqrt(this->_n_clusters)));
        balanced_iteration_stats.fineclustering_iteration_stats.clear();
        balanced_iteration_stats.mesoclustering_iteration_stats.clear();
        balanced_iteration_stats.refinement_iteration_stats.clear();
        if (n < this->_n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters"
            );
        }
        if (n_queries > 0 && queries == nullptr && !this->_config.sample_queries) {
            throw std::invalid_argument(
                "Queries must be provided if n_queries > 0 and sample_queries is false"
            );
        }
        const vector_value_t* SKM_RESTRICT data_p = data;
        this->_n_samples = this->GetNVectorsToSample(n, this->_n_clusters);
        if (this->_n_samples < this->_n_clusters) {
            throw std::runtime_error(
                "Not enough samples to train. Try increasing the sampling_fraction or "
                "max_points_per_cluster"
            );
        }
        {
            SKM_PROFILE_SCOPE("allocator");
            this->mesoclusters_assignments.resize(n);
            this->final_assignments.resize(n);
            // We use the same buffers for the mesocentroids
            this->_centroids.resize(this->_n_clusters * this->_d);
            this->_horizontal_centroids.resize(this->_n_clusters * this->_d);
            this->_prev_centroids.resize(this->_n_clusters * this->_d);
            this->_cluster_sizes.resize(this->_n_clusters);
            this->_assignments.resize(n);
            this->_distances.resize(n);
            this->_data_norms.resize(this->_n_samples);
            this->_centroid_norms.resize(this->_n_clusters);
        }
        std::vector<distance_t> tmp_distances_buf(X_BATCH_SIZE * Y_BATCH_SIZE);
        this->_vertical_d = PDXLayout<q, alpha>::GetDimensionSplit(this->_d).vertical_d;
        this->_partial_horizontal_centroids.resize(this->_n_clusters * this->_vertical_d);

        this->_partial_d = std::max<uint32_t>(MIN_PARTIAL_D, this->_vertical_d / 2);
        if (this->_partial_d > this->_vertical_d) {
            this->_partial_d = this->_vertical_d;
        }
        if (this->_config.verbose) {
            std::cout << "Front dimensions (d') = " << this->_partial_d << std::endl;
            std::cout << "Trailing dimensions (d'') = " << this->_d - this->_vertical_d
                      << std::endl;
        }

        //
        // MESOCLUSTERING
        //
        auto centroids_pdx_wrapper =
            this->GenerateCentroids(data_p, this->_n_samples, n_mesoclusters);
        if (this->_config.verbose) {
            std::cout << "Sampling data..." << std::endl;
        }
        std::vector<vector_value_t>
            data_samples_buffer; // Samples for mesoclustering and fineclustering
        this->SampleAndRotateVectors(data_p, data_samples_buffer, n, this->_n_samples, true);
        auto data_to_cluster = data_samples_buffer.data();

        {
            SKM_PROFILE_SCOPE("rotator");
            if (this->_config.verbose)
                std::cout << "Rotating..." << std::endl;
            this->_pruner->Rotate(
                this->_horizontal_centroids.data(), this->_prev_centroids.data(), n_mesoclusters
            );
        }

        this->GetL2NormsRowMajor(data_to_cluster, this->_n_samples, this->_data_norms.data());
        this->GetL2NormsRowMajor(
            this->_prev_centroids.data(), n_mesoclusters, this->_centroid_norms.data()
        );

        // TODO(@lkuffo, crit) N_QUERIES LOGIC

        if (this->_config.verbose)
            std::cout << "1st iteration..." << std::endl;

        size_t iter_idx = 0;
        float best_recall = 0.0f;
        size_t iters_without_improvement = 0;

        // Buffers for RunIteration (needed for function signature even if unused in GEMM-only mode)
        std::vector<vector_value_t> centroids_partial_norms(n_mesoclusters);
        std::vector<size_t> not_pruned_counts(this->_n_samples);

        this->RunIteration<true>(
            data_to_cluster,
            tmp_distances_buf.data(),
            centroids_pdx_wrapper,
            centroids_partial_norms,
            not_pruned_counts,
            nullptr,  // rotated_queries (TODO: implement query logic)
            0,        // n_queries
            this->_n_samples,
            n_mesoclusters,
            iter_idx,
            true,  // is_first_iter
            this->balanced_iteration_stats.mesoclustering_iteration_stats
        );

        iter_idx = 1;
        best_recall = this->_recall;

        // My theory is that for Mesoclusters SuperKMeans is not going to work as well,
        // Depending on how many clusters we have
        // But for fineclusters, since the ratio is around 1:100, then it should work.
        if (this->balanced_config.iters_mesoclustering > 1) {
            //
            // FULL GEMM on low-dimensional data or too few clusters
            //
            if (this->_d < DIMENSION_THRESHOLD_FOR_PRUNING ||
                n_mesoclusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                for (; iter_idx < this->balanced_config.iters_mesoclustering; ++iter_idx) {
                    this->RunIteration<true>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        centroids_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr,  // rotated_queries
                        0,        // n_queries
                        this->_n_samples,
                        n_mesoclusters,
                        iter_idx,
                        false,  // is_first_iter
                        this->balanced_iteration_stats.mesoclustering_iteration_stats
                    );
                    if (this->_config.early_termination &&
                        this->ShouldStopEarly(
                            n_queries > 0, best_recall, iters_without_improvement, iter_idx
                        )) {
                        break;
                    }
                }
            } else { // Rest of Iterations with GEMM+PRUNING
                this->GetPartialL2NormsRowMajor(
                    data_to_cluster, this->_n_samples, this->_data_norms.data(), this->_partial_d
                );
                for (; iter_idx < this->balanced_config.iters_mesoclustering; ++iter_idx) {
                    this->RunIteration<false>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        centroids_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr,  // rotated_queries
                        0,        // n_queries
                        this->_n_samples,
                        n_mesoclusters,
                        iter_idx,
                        false,  // is_first_iter
                        this->balanced_iteration_stats.mesoclustering_iteration_stats
                    );
                    if (this->_config.early_termination &&
                        this->ShouldStopEarly(
                            n_queries > 0, best_recall, iters_without_improvement, iter_idx
                        )) {
                        break;
                    }
                }
            }
        }
        auto meso_centroids = this->GetOutputCentroids(true);
        mesoclusters_assignments = this->_assignments; // TODO(@lkuffo, high): Deep copy?
        // TODO(@lkuffo, high: Need a fast assign, for now we use the iter_idx-1 assignments)
        // mesoclusters_assignments = Assign(data, meso_centroids.data(), n, n_mesoclusters);

        //
        // FINE-CLUSTERING
        // Each mesocluster is re-clustered sequentially
        //
        size_t max_mesocluster_size = *std::max_element(this->_cluster_sizes.begin(), this->_cluster_sizes.end());
        std::vector<vector_value_t> mesocluster_buffer(max_mesocluster_size * this->_d);
        std::vector<vector_value_t> mesocluster_data_norms(max_mesocluster_size);
        std::vector<vector_value_t> mesocluster_partial_data_norms(max_mesocluster_size);
        size_t fineclusters_left_to_build = this->_n_clusters;
        for (size_t k = 0; k < n_mesoclusters; ++k) {
            size_t n_fineclusters = std::min(n_mesoclusters, fineclusters_left_to_build);
            fineclusters_left_to_build -= n_fineclusters;

            // Reset partial_d
            // Copy in mesocluster_buffer, the vectors assigned to this mesocluster
            // Get how many they are (n_samples_in_mesocluster)
            // They are already rotated, so you just have to copy
            // Copy the full norms of these vectors
            // DO NOT SAMPLE POINTS... SAMPLE WAS ALREADY TAKEN
            // Generate centroids from this buffer

            // Run the clustering as normal. Everything should fall in place

            // Get output_centroids
            // Copy output_centroids into a final buffer with the final centroids
            // Copy n_samples_in_mesocluster of this->_assignments into final_assignments
            // final_assignments_offset += n_samples_in_mesocluster
        }

        // Copy final_assignments into this->_assignments
        // 2 refinement iterations GEMM+PRUNING with data_to_sample and final_centroids
        // and all n_clusters
        //

        this->_trained = true;
        auto output_centroids = this->GetOutputCentroids(this->_config.unrotate_centroids);
        if (this->_config.perform_assignments) {
            this->_assignments = Assign(data, output_centroids.data(), n, this->_n_clusters);
        }
        if (this->_config.verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }

    size_t n_mesoclusters;

    std::vector<uint32_t> mesoclusters_assignments;
    std::vector<uint32_t> final_assignments;
    std::vector<centroid_value_t> meso_centroids;
    BalancedSuperKMeansConfig balanced_config;
    BalancedSuperKMeansIterationStats balanced_iteration_stats;
};

} // namespace skmeans
