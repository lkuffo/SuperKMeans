#pragma once

#include "superkmeans/superkmeans.h"

namespace skmeans {

/**
 * @brief Configuration parameters for Balanced SuperKMeans clustering.
 */
struct BalancedSuperKMeansConfig : SuperKMeansConfig {
    uint32_t n_points_per_mesocluster = 1024;
    uint32_t iters_mesoclustering = 10;
    uint32_t iters_fineclustering = 10;
    uint32_t iters_refinement = 2;
};

/**
 * @brief Statistics for BalancedSuperKMeans clustering.
 */
struct BalancedSuperKMeansIterationStats {
    std::vector<SuperKMeansIterationStats> mesoclustering_iteration_stats;
    std::vector<SuperKMeansIterationStats> refinement_iteration_stats;
    std::vector<SuperKMeansIterationStats> fineclustering_iteration_stats;
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
        n_mesoclusters = static_cast<size_t>(std::round(std::sqrt(this->_n_clusters)));
        balanced_iteration_stats.fineclustering_iteration_stats.clear();
        balanced_iteration_stats.mesoclustering_iteration_stats.clear();
        balanced_iteration_stats.refinement_iteration_stats.clear();
        if (n < this->_n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters"
            );
        }
        if (n_queries > 0 && queries == nullptr && !this->balanced_config.sample_queries) {
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
            this->mesoclusters_assignments.resize(this->_n_samples);
            this->final_assignments.resize(this->_n_samples);
            this->mesoclusters_sizes.resize(n_mesoclusters);
            this->final_centroids.resize(this->_n_clusters * this->_d);
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
        auto initial_partial_d = this->_partial_d;
        if (this->_partial_d > this->_vertical_d) {
            this->_partial_d = this->_vertical_d;
        }
        if (this->balanced_config.verbose) {
            std::cout << "Front dimensions (d') = " << this->_partial_d << std::endl;
            std::cout << "Trailing dimensions (d'') = " << this->_d - this->_vertical_d
                      << std::endl;
        }

        //
        // MESOCLUSTERING
        //
        if (this->balanced_config.verbose) {
            std::cout << "\n=== PHASE 1: MESOCLUSTERING (k=" << n_mesoclusters << " clusters) ===" << std::endl;
        }
        auto centroids_pdx_wrapper =
            this->GenerateCentroids(data_p, this->_n_samples, n_mesoclusters);
        if (this->balanced_config.verbose) {
            std::cout << "Sampling data..." << std::endl;
        }
        std::vector<vector_value_t>
            data_samples_buffer; // Samples for mesoclustering and fineclustering
        this->SampleAndRotateVectors(data_p, data_samples_buffer, n, this->_n_samples, true);
        auto data_to_cluster = data_samples_buffer.data();
        auto initial_n_samples = this->_n_samples;

        {
            SKM_PROFILE_SCOPE("rotator");
            if (this->balanced_config.verbose)
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

        if (this->balanced_config.verbose)
            std::cout << "1st iteration..." << std::endl;

        size_t iter_idx = 0;
        float best_recall = 0.0f;
        size_t iters_without_improvement = 0;

        // Buffers for RunIteration (needed for function signature even if unused in GEMM-only mode)
        std::vector<vector_value_t> centroids_partial_norms(this->_n_clusters);
        std::vector<size_t> not_pruned_counts(this->_n_samples);

        this->template RunIteration<true>(
            data_to_cluster,
            tmp_distances_buf.data(),
            centroids_pdx_wrapper,
            centroids_partial_norms,
            not_pruned_counts,
            nullptr, // rotated_queries (TODO: implement query logic)
            0,       // n_queries
            this->_n_samples,
            n_mesoclusters,
            iter_idx,
            true, // is_first_iter
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
                    this->template RunIteration<true>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        centroids_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr, // rotated_queries
                        0,       // n_queries
                        this->_n_samples,
                        n_mesoclusters,
                        iter_idx,
                        false, // is_first_iter
                        this->balanced_iteration_stats.mesoclustering_iteration_stats
                    );
                    if (this->balanced_config.early_termination &&
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
                    this->template RunIteration<false>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        centroids_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr, // rotated_queries
                        0,       // n_queries
                        this->_n_samples,
                        n_mesoclusters,
                        iter_idx,
                        false, // is_first_iter
                        this->balanced_iteration_stats.mesoclustering_iteration_stats
                    );
                    if (this->balanced_config.early_termination &&
                        this->ShouldStopEarly(
                            n_queries > 0, best_recall, iters_without_improvement, iter_idx
                        )) {
                        break;
                    }
                }
            }
        }
        auto meso_centroids = this->GetOutputCentroids(false);
        mesoclusters_sizes = this->_cluster_sizes; // TODO(@lkuffo, high): Deep copy?
        mesoclusters_assignments =
            this->_assignments; // TODO(@lkuffo, high): Deep copy? I dont really like this because
                                // we need smaller buffers (we are taking samples)
        immutable_data_norms = this->_data_norms; // TODO(@lkuffo, high): Deep copy?
        // TODO(@lkuffo, high: Need a fast assign, for now we use the iter_idx-1 assignments)
        // mesoclusters_assignments = Assign(data, meso_centroids.data(), n, n_mesoclusters);

        //
        // FINE-CLUSTERING
        // Each mesocluster is re-clustered sequentially
        //
        if (this->balanced_config.verbose) {
            std::cout << "\n=== PHASE 2: FINE-CLUSTERING (subdividing " << n_mesoclusters
                      << " mesoclusters into total " << this->_n_clusters << " clusters) ===" << std::endl;
        }
        size_t max_mesocluster_size =
            *std::max_element(this->_cluster_sizes.begin(), this->_cluster_sizes.end());
        std::vector<vector_value_t> mesocluster_buffer(max_mesocluster_size * this->_d);
        std::vector<vector_value_t> mesocluster_data_norms(max_mesocluster_size);
        std::vector<vector_value_t> mesocluster_partial_data_norms(max_mesocluster_size);
        std::vector<uint32_t> assignments_indirection_buffer(max_mesocluster_size);
        size_t fineclusters_left_to_build = this->_n_clusters;
        size_t fineclusters_offset = 0;
        for (size_t k = 0; k < n_mesoclusters; ++k) {
            size_t n_fineclusters =
                (k == n_mesoclusters - 1) ? fineclusters_left_to_build : n_mesoclusters;
            fineclusters_left_to_build -= n_fineclusters;
            this->_partial_d = initial_partial_d;

            auto mesocluster_size = mesoclusters_sizes[k];
            this->_n_samples = mesocluster_size;
            CompactMesoclusterToBuffer(
                k,
                mesocluster_size,
                data_to_cluster,
                initial_n_samples,
                mesocluster_buffer.data(),
                assignments_indirection_buffer.data()
            );
            auto mesocluster_data_to_cluster = mesocluster_buffer.data();
            auto mesocluster_centroids_pdx_wrapper = this->GenerateCentroids(
                mesocluster_data_to_cluster, mesocluster_size, n_fineclusters, false
            );
            this->GetL2NormsRowMajor(
                this->_horizontal_centroids.data(), n_fineclusters, this->_centroid_norms.data()
            );

            size_t fine_iter_idx = 0;
            float fine_best_recall = 0.0f;
            iters_without_improvement = 0;

            this->template RunIteration<true>(
                mesocluster_data_to_cluster,
                tmp_distances_buf.data(),
                mesocluster_centroids_pdx_wrapper,
                centroids_partial_norms,
                not_pruned_counts,
                nullptr, // rotated_queries (TODO: implement query logic)
                0,       // n_queries
                this->_n_samples,
                n_fineclusters,
                fine_iter_idx,
                true, // is_first_iter
                this->balanced_iteration_stats.fineclustering_iteration_stats
            );

            fine_iter_idx = 1;
            fine_best_recall = this->_recall;

            if (this->balanced_config.iters_fineclustering > 1) {
                //
                // FULL GEMM on low-dimensional data or too few clusters
                //
                if (this->_d < DIMENSION_THRESHOLD_FOR_PRUNING ||
                    n_fineclusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                    for (; fine_iter_idx < this->balanced_config.iters_fineclustering;
                         ++fine_iter_idx) {
                        this->template RunIteration<true>(
                            mesocluster_data_to_cluster,
                            tmp_distances_buf.data(),
                            mesocluster_centroids_pdx_wrapper,
                            centroids_partial_norms,
                            not_pruned_counts,
                            nullptr, // rotated_queries
                            0,       // n_queries
                            this->_n_samples,
                            n_fineclusters,
                            fine_iter_idx,
                            false, // is_first_iter
                            this->balanced_iteration_stats.fineclustering_iteration_stats
                        );
                        if (this->balanced_config.early_termination && this->ShouldStopEarly(
                                                                   n_queries > 0,
                                                                   fine_best_recall,
                                                                   iters_without_improvement,
                                                                   fine_iter_idx
                                                               )) {
                            break;
                        }
                    }
                } else { // Rest of Iterations with GEMM+PRUNING
                    this->GetPartialL2NormsRowMajor(
                        mesocluster_data_to_cluster,
                        this->_n_samples,
                        this->_data_norms.data(),
                        this->_partial_d
                    );
                    for (; fine_iter_idx < this->balanced_config.iters_fineclustering;
                         ++fine_iter_idx) {
                        this->template RunIteration<false>(
                            mesocluster_data_to_cluster,
                            tmp_distances_buf.data(),
                            mesocluster_centroids_pdx_wrapper,
                            centroids_partial_norms,
                            not_pruned_counts,
                            nullptr, // rotated_queries
                            0,       // n_queries
                            this->_n_samples,
                            n_fineclusters,
                            fine_iter_idx,
                            false, // is_first_iter
                            this->balanced_iteration_stats.fineclustering_iteration_stats
                        );
                        if (this->balanced_config.early_termination && this->ShouldStopEarly(
                                                                   n_queries > 0,
                                                                   fine_best_recall,
                                                                   iters_without_improvement,
                                                                   fine_iter_idx
                                                               )) {
                            break;
                        }
                    }
                }
            }

            auto fine_centroids = this->GetOutputCentroids(false);
            // If we want to use PRUNING in the refinement iterations, I need this indirection to be
            // resolved
            GetTrueAssignmentsFromIndirectionBuffer(
                assignments_indirection_buffer.data(),
                final_assignments.data(),
                this->_assignments.data(),
                mesocluster_size,
                fineclusters_offset
            );
            // We move the resulting centroids from this fineclustering to the final buffer of
            // centroids
            memcpy(
                static_cast<void*>(final_centroids.data() + fineclusters_offset * this->_d),
                static_cast<void*>(fine_centroids.data()),
                sizeof(centroid_value_t) * n_fineclusters * this->_d
            );
            fineclusters_offset += n_fineclusters;
        }

        // Now we move to the last refinement phase in which we perform clustering with all
        // _n_clusters. Recall our initial buffers for centroids have enough space for _n_clusters.
        if (this->balanced_config.verbose) {
            std::cout << "\n=== PHASE 3: REFINEMENT (fine-tuning all " << this->_n_clusters
                      << " clusters) ===" << std::endl;
        }
        this->_n_samples = initial_n_samples;
        this->_partial_d = initial_partial_d;

        // We just transfer the state of centroids to the proper class variables, no rotation.
        auto final_refinement_pdx_wrapper =
            SetupCentroids(final_centroids.data(), this->_n_clusters);
        this->GetL2NormsRowMajor(
            this->_horizontal_centroids.data(), this->_n_clusters, this->_centroid_norms.data()
        );

        // RunIteration<false>
        // 2 refinement iterations GEMM+PRUNING with data_to_sample and final_centroids
        size_t refinement_iter_idx = 0;
        if (this->balanced_config.iters_refinement > 0) {
            //
            // FULL GEMM on low-dimensional data or too few clusters
            //
            if (this->_d < DIMENSION_THRESHOLD_FOR_PRUNING ||
                this->_n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                for (; refinement_iter_idx < this->balanced_config.iters_refinement;
                     ++refinement_iter_idx) {
                    this->template RunIteration<true>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        final_refinement_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr, // rotated_queries
                        0,       // n_queries
                        this->_n_samples,
                        this->_n_clusters,
                        refinement_iter_idx,
                        false, // is_first_iter
                        this->balanced_iteration_stats.refinement_iteration_stats
                    );
                }
            } else { // Rest of Iterations with GEMM+PRUNING
                // TODO(@lkuffo, high): The only reason I need to do this (again) in the first time
                //   is because we are
                //   using the same this->_data_norms.data() buffer in the fineclustering, which
                //   replaces the norms that I already calculated before and put in this buffer.
                this->GetPartialL2NormsRowMajor(
                    data_to_cluster, this->_n_samples, this->_data_norms.data(), this->_partial_d
                );
                for (; refinement_iter_idx < this->balanced_config.iters_refinement;
                     ++refinement_iter_idx) {
                    this->template RunIteration<false>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        final_refinement_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr, // rotated_queries
                        0,       // n_queries
                        this->_n_samples,
                        this->_n_clusters,
                        refinement_iter_idx,
                        false, // is_first_iter
                        this->balanced_iteration_stats.refinement_iteration_stats
                    );
                }
            }
        }

        this->_trained = true;
        auto output_centroids = this->GetOutputCentroids(this->balanced_config.unrotate_centroids);
        if (this->balanced_config.perform_assignments) {
            this->_assignments = this->Assign(data, output_centroids.data(), n, this->_n_clusters);
        }
        if (this->balanced_config.verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }

    /*
     * Compact data assigned to a mesocluster in mesocluster_buffer
     * They are already rotated, so we just have to copy
     * Additionally, we have to copy their norms in a sequential buffer to not recompute them
     * In theory, we don't have to re-sample data here.
     */
    void CompactMesoclusterToBuffer(
        const size_t mesocluster_id,
        const size_t mesocluster_size,
        const vector_value_t* SKM_RESTRICT data,
        const size_t n_samples,
        vector_value_t* SKM_RESTRICT mesocluster_buffer,
        uint32_t* SKM_RESTRICT assignments_indirection_buffer
    ) {
        size_t samples_compacted = 0;
        // Iterate through all data to compact the mesocluster in a contiguous block
        // TODO(@lkuffo, high): I would need to see how numpy does this to be more efficient
        for (size_t i = 0; i < n_samples; ++i) {
            if (mesoclusters_assignments[i] == mesocluster_id) {
                this->_data_norms[samples_compacted] = immutable_data_norms[i];
                assignments_indirection_buffer[samples_compacted] = i;
                memcpy(
                    static_cast<void*>(mesocluster_buffer + samples_compacted * this->_d),
                    static_cast<const void*>(data + i * this->_d),
                    sizeof(vector_value_t) * this->_d
                );
                samples_compacted++;
            }
        }
    }

    void GetTrueAssignmentsFromIndirectionBuffer(
        const uint32_t* SKM_RESTRICT assignments_indirection_buffer,
        uint32_t* SKM_RESTRICT output_assignments,
        const uint32_t* SKM_RESTRICT input_assignments,
        const size_t n_samples_in_mesocluster,
        const size_t cluster_id_offset
    ) {
        for (size_t i = 0; i < n_samples_in_mesocluster; ++i) {
            size_t original_idx = assignments_indirection_buffer[i];
            uint32_t local_cluster_id = input_assignments[i];
            // Assignments in fineclustering are from 0 to n_fineclusters-1
            // We need to add the offset to put the global cluster id in the final assignments
            uint32_t global_cluster_id = local_cluster_id + cluster_id_offset;
            output_assignments[original_idx] = global_cluster_id;
        }
    }

    /**
     * @brief Setup centroids to be used for clustering
     *
     *
     * @param centroids Centroids to setup
     * @param n_clusters Number of centroids to setupt
     * @return PDXLayout wrapper for the centroids
     */
    PDXLayout<q, alpha> SetupCentroids(
        const centroid_value_t* SKM_RESTRICT centroids,
        const size_t n_clusters
    ) {
        memcpy(
            (void*) (this->_horizontal_centroids.data()),
            (void*) (centroids),
            sizeof(centroid_value_t) * n_clusters * this->_d
        );
        {
            SKM_PROFILE_SCOPE("consolidate/pdxify");
            PDXLayout<q, alpha>::template PDXify<false>(
                this->_horizontal_centroids.data(), this->_centroids.data(), n_clusters, this->_d
            );
        }
        //! We wrap _centroids and _partial_horizontal_centroids in the PDXLayout wrapper
        //! Any updates to these objects is reflected in the PDXLayout
        auto pdx_centroids = PDXLayout<q, alpha>(
            this->_centroids.data(),
            *this->_pruner,
            n_clusters,
            this->_d,
            this->_partial_horizontal_centroids.data()
        );
        this->CentroidsToAuxiliaryHorizontal(n_clusters);
        return pdx_centroids;
    }

    size_t n_mesoclusters = 0;

    std::vector<uint32_t> mesoclusters_assignments;
    std::vector<uint32_t> mesoclusters_sizes;
    std::vector<uint32_t> final_assignments;
    std::vector<vector_value_t> immutable_data_norms;
    std::vector<centroid_value_t> meso_centroids;
    std::vector<centroid_value_t> final_centroids;
    BalancedSuperKMeansConfig balanced_config;
    BalancedSuperKMeansIterationStats balanced_iteration_stats;
};

} // namespace skmeans
