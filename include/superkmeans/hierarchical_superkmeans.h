#pragma once

#include "superkmeans/pdx/utils.h"
#include "superkmeans/superkmeans.h"

namespace skmeans {

/**
 * @brief Configuration parameters for Hierarchical SuperKMeans clustering.
 */
struct HierarchicalSuperKMeansConfig : SuperKMeansConfig {
    uint32_t iters_mesoclustering = 3;
    uint32_t iters_fineclustering = 5;
    uint32_t iters_refinement = 1;
};

/**
 * @brief Statistics for HierarchicalSuperKMeans clustering.
 */
struct HierarchicalSuperKMeansIterationStats {
    std::vector<SuperKMeansIterationStats> mesoclustering_iteration_stats;
    std::vector<SuperKMeansIterationStats> refinement_iteration_stats;
    std::vector<SuperKMeansIterationStats> fineclustering_iteration_stats;
};

template <Quantization q = Quantization::f32, DistanceFunction alpha = DistanceFunction::l2>
class HierarchicalSuperKMeans : public SuperKMeans<q, alpha> {
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
    HierarchicalSuperKMeans(
        size_t n_clusters,
        size_t dimensionality,
        const HierarchicalSuperKMeansConfig& config
    )
        : SuperKMeans<q, alpha>(n_clusters, dimensionality, config), hierarchical_config(config) {
        this->_pruner = std::make_unique<pruner_t>(
            dimensionality, HIERARCHICAL_PRUNER_INITIAL_THRESHOLD, this->_config.seed
        );
        SKMEANS_ENSURE_POSITIVE(config.iters_mesoclustering);
        SKMEANS_ENSURE_POSITIVE(config.iters_fineclustering);

        static_cast<SuperKMeansConfig&>(hierarchical_config) = this->_config;

        if (n_clusters <= 128) {
            std::cout
                << "WARNING: n_clusters <= 128 is not recommended for HierarchicalSuperKMeans. "
                   "Consider using at least 128 clusters."
                << std::endl;
        }
    }

    /**
     * @brief Default constructor
     */
    HierarchicalSuperKMeans(size_t n_clusters, size_t dimensionality)
        : HierarchicalSuperKMeans(n_clusters, dimensionality, HierarchicalSuperKMeansConfig{}) {}

    /**
     * @brief Run hierarchical k-means clustering to determine centroids
     * We don't support Early Termination by Recall here.
     * queries and n_queries are ignored. But we keep the function signature for compatibility.
     *
     * @param data Pointer to the data matrix (row-major, n × d)
     * @param n Number of points (rows) in the data matrix
     *
     * @return std::vector<skmeans_centroid_value_t<q>> Trained centroids
     */
    std::vector<skmeans_centroid_value_t<q>> Train(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        const vector_value_t* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0
    ) {
        SKMEANS_ENSURE_POSITIVE(n);
        if (this->_trained) {
            throw std::runtime_error("The clustering has already been trained");
        }
        if (n_queries > 0) {
            std::cout << "WARNING: Early Termination by Recall is not supported in "
                         "HierarchicalSuperKMeans"
                      << std::endl;
        }
        n_mesoclusters = GetNMesoclusters(this->_n_clusters);
        hierarchical_iteration_stats.fineclustering_iteration_stats.clear();
        hierarchical_iteration_stats.mesoclustering_iteration_stats.clear();
        hierarchical_iteration_stats.refinement_iteration_stats.clear();
        if (n < this->_n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters"
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
            // Buffers to concatenate each fineclustering assignments and centroids
            this->final_assignments.resize(this->_n_samples);
            this->final_centroids.resize(this->_n_clusters * this->_d);
            // These buffers are reused for all three phases
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
        auto initial_partial_d = this->_partial_d;
        
        if (this->hierarchical_config.verbose) {
            std::cout << "Front dimensions (d') = " << this->_partial_d << std::endl;
            std::cout << "Trailing dimensions (d'') = " << this->_d - this->_vertical_d
                      << std::endl;
        }

        //
        // MESOCLUSTERING
        //
        TicToc timer_mesoclustering;
        timer_mesoclustering.Tic();
        if (this->hierarchical_config.verbose) {
            std::cout << "\n=== PHASE 1: MESOCLUSTERING (k=" << n_mesoclusters
                      << " clusters) ===" << std::endl;
        }
        auto centroids_pdx_wrapper = this->GenerateCentroids(
            data_p,
            this->_n_samples,
            n_mesoclusters,
            !this->hierarchical_config.data_already_rotated
        );
        if (this->hierarchical_config.verbose) {
            std::cout << "Sampling data..." << std::endl;
        }
        // Samples for both mesoclustering and fineclustering
        std::vector<vector_value_t> data_samples_buffer;
        this->SampleAndRotateVectors(
            data_p,
            data_samples_buffer,
            n,
            this->_n_samples,
            !this->hierarchical_config.data_already_rotated
        );
        auto data_to_cluster = data_samples_buffer.data();
        auto initial_n_samples = this->_n_samples;
        this->RotateOrCopy(
            this->_horizontal_centroids.data(),
            this->_prev_centroids.data(),
            n_mesoclusters,
            !this->hierarchical_config.data_already_rotated
        );
        this->GetL2NormsRowMajor(data_to_cluster, this->_n_samples, this->_data_norms.data());
        this->GetL2NormsRowMajor(
            this->_prev_centroids.data(), n_mesoclusters, this->_centroid_norms.data()
        );

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
            nullptr, // queries
            0,       // n_queries
            this->_n_samples,
            n_mesoclusters,
            iter_idx,
            true, // is_first_iter
            this->hierarchical_iteration_stats.mesoclustering_iteration_stats
        );

        iter_idx = 1;
        best_recall = this->_recall;

        // Save full norms before potentially computing partial norms
        // (needed because GEMM+PRUNING path will overwrite _data_norms with partial norms)
        {
            SKM_PROFILE_SCOPE("allocator");
            immutable_data_norms = this->_data_norms;
        }

        if (this->hierarchical_config.iters_mesoclustering > 1) {
            //
            // FULL GEMM on low-dimensional data or too few clusters
            // or first 2 iterations
            //
            bool partial_norms_computed = false;
            for (; iter_idx < this->hierarchical_config.iters_mesoclustering; ++iter_idx) {
                if (iter_idx < 1 || this->_d < DIMENSION_THRESHOLD_FOR_PRUNING ||
                    n_mesoclusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                    this->template RunIteration<true>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        centroids_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr, // queries
                        0,       // n_queries
                        this->_n_samples,
                        n_mesoclusters,
                        iter_idx,
                        false, // is_first_iter
                        this->hierarchical_iteration_stats.mesoclustering_iteration_stats
                    );
                } else { // Rest of Iterations with GEMM+PRUNING
                    if (!partial_norms_computed) {
                        this->GetPartialL2NormsRowMajor(
                            data_to_cluster,
                            this->_n_samples,
                            this->_data_norms.data(),
                            this->_partial_d
                        );
                        partial_norms_computed = true;
                    }
                    this->template RunIteration<false>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        centroids_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr, // queries
                        0,       // n_queries
                        this->_n_samples,
                        n_mesoclusters,
                        iter_idx,
                        false, // is_first_iter
                        this->hierarchical_iteration_stats.mesoclustering_iteration_stats
                    );
                }
                if (this->hierarchical_config.early_termination &&
                    this->ShouldStopEarly(
                        false, best_recall, iters_without_improvement, iter_idx
                    )) {
                    break;
                }
            }
        }
        timer_mesoclustering.Toc();

        {
            SKM_PROFILE_SCOPE("allocator");
            mesoclusters_sizes.assign(
                this->_cluster_sizes.begin(), this->_cluster_sizes.begin() + n_mesoclusters
            );
            mesoclusters_assignments.assign(
                this->_assignments.begin(), this->_assignments.begin() + this->_n_samples
            );
        }

        // Build partitioned index for efficient mesocluster compaction
        std::vector<size_t> mesocluster_indices_flat(this->_n_samples);
        std::vector<size_t> mesocluster_offsets(n_mesoclusters + 1);
        {
            SKM_PROFILE_SCOPE("compact_indices");
            mesocluster_offsets[0] = 0;
            for (size_t k = 0; k < n_mesoclusters; ++k) {
                mesocluster_offsets[k + 1] = mesocluster_offsets[k] + mesoclusters_sizes[k];
            }
            std::vector<size_t> next_to_write_index = mesocluster_offsets;
            for (size_t i = 0; i < this->_n_samples; ++i) {
                size_t cluster_id = mesoclusters_assignments[i];
                mesocluster_indices_flat[next_to_write_index[cluster_id]++] = i;
            }
        }

        //
        // FINE-CLUSTERING
        // Each mesocluster is re-clustered sequentially
        // Potential improvement: Doing 2 only-GEMM iterations and delegate the rest to GEMM+PRUNING
        // seems an interesting idea. However, we need to evaluate this with a larger dataset (+100M
        // vectors)
        //
        if (this->hierarchical_config.verbose) {
            std::cout << "\n=== PHASE 2: FINE-CLUSTERING (subdividing " << n_mesoclusters
                      << " mesoclusters into total " << this->_n_clusters
                      << " clusters) ===" << std::endl;
        }
        // Calculate proportional allocation of fine clusters per mesocluster
        auto fine_clusters_nums = ArrangeFineClusters(
            this->_n_clusters, n_mesoclusters, this->_n_samples, mesoclusters_sizes
        );

        TicToc timer_fineclustering;
        timer_fineclustering.Tic();

        size_t max_mesocluster_size = *std::max_element(
            this->_cluster_sizes.begin(), this->_cluster_sizes.begin() + n_mesoclusters
        );
        std::vector<vector_value_t> mesocluster_buffer(max_mesocluster_size * this->_d);
        std::vector<uint32_t> assignments_indirection_buffer(max_mesocluster_size);
        size_t fineclusters_offset = 0;
        for (size_t k = 0; k < n_mesoclusters; ++k) {
            size_t n_fineclusters = fine_clusters_nums[k];
            if (n_fineclusters == 0) {
                continue;
            }
            this->_partial_d = initial_partial_d;

            auto mesocluster_size = mesoclusters_sizes[k];
            // auto points_per_finecluster = static_cast<float>(mesocluster_size) /
            // static_cast<float>(n_fineclusters);
            std::cout << "n_fineclusters = " << n_fineclusters << std::endl;
            this->_n_samples = mesocluster_size;
            CompactMesoclusterToBuffer(
                mesocluster_size,
                data_to_cluster,
                mesocluster_buffer.data(),
                assignments_indirection_buffer.data(),
                mesocluster_indices_flat.data() + mesocluster_offsets[k]
            );
            auto mesocluster_data_to_cluster = mesocluster_buffer.data();
            auto mesocluster_centroids_pdx_wrapper = this->GenerateCentroids(
                mesocluster_data_to_cluster, mesocluster_size, n_fineclusters, false
            );
            // Copy centroids to _prev_centroids for use in the first RunIteration
            // (is_first_iter=true skips the swap, so _prev_centroids must be populated)
            memcpy(
                this->_prev_centroids.data(),
                this->_horizontal_centroids.data(),
                sizeof(centroid_value_t) * n_fineclusters * this->_d
            );
            this->GetL2NormsRowMajor(
                this->_prev_centroids.data(), n_fineclusters, this->_centroid_norms.data()
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
                nullptr, // queries
                0,       // n_queries
                this->_n_samples,
                n_fineclusters,
                fine_iter_idx,
                true, // is_first_iter
                this->hierarchical_iteration_stats.fineclustering_iteration_stats
            );

            fine_iter_idx = 1;
            fine_best_recall = this->_recall;

            if (this->hierarchical_config.iters_fineclustering > 1) {
                //
                // FULL GEMM on low-dimensional data or too few clusters
                // or first 2 iterations
                //
                bool partial_norms_computed = false;
                for (; fine_iter_idx < this->hierarchical_config.iters_fineclustering;
                     ++fine_iter_idx) {
                    if (fine_iter_idx < 1 || this->_d < DIMENSION_THRESHOLD_FOR_PRUNING ||
                        n_fineclusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                        this->template RunIteration<true>(
                            mesocluster_data_to_cluster,
                            tmp_distances_buf.data(),
                            mesocluster_centroids_pdx_wrapper,
                            centroids_partial_norms,
                            not_pruned_counts,
                            nullptr, // queries
                            0,       // n_queries
                            this->_n_samples,
                            n_fineclusters,
                            fine_iter_idx,
                            false, // is_first_iter
                            this->hierarchical_iteration_stats.fineclustering_iteration_stats
                        );
                    } else { // Rest of Iterations with GEMM+PRUNING
                        if (!partial_norms_computed) {
                            this->GetPartialL2NormsRowMajor(
                                mesocluster_data_to_cluster,
                                this->_n_samples,
                                this->_data_norms.data(),
                                this->_partial_d
                            );
                            partial_norms_computed = true;
                        }
                        this->template RunIteration<false>(
                            mesocluster_data_to_cluster,
                            tmp_distances_buf.data(),
                            mesocluster_centroids_pdx_wrapper,
                            centroids_partial_norms,
                            not_pruned_counts,
                            nullptr, // queries
                            0,       // n_queries
                            this->_n_samples,
                            n_fineclusters,
                            fine_iter_idx,
                            false, // is_first_iter
                            this->hierarchical_iteration_stats.fineclustering_iteration_stats
                        );
                    }
                    if (this->hierarchical_config.early_termination &&
                        this->ShouldStopEarly(
                            false, fine_best_recall, iters_without_improvement, fine_iter_idx
                        )) {
                        break;
                    }
                }
            }

            GetTrueAssignmentsFromIndirectionBuffer(
                assignments_indirection_buffer.data(),
                final_assignments.data(),
                this->_assignments.data(),
                mesocluster_size,
                fineclusters_offset
            );

            // We move the resulting centroids from this fineclustering to the final buffer of
            // centroids
            {
                SKM_PROFILE_SCOPE("copy_fine_centroids");
                memcpy(
                    static_cast<void*>(final_centroids.data() + fineclusters_offset * this->_d),
                    static_cast<void*>(this->_horizontal_centroids.data()),
                    sizeof(centroid_value_t) * n_fineclusters * this->_d
                );
            }
            fineclusters_offset += n_fineclusters;
        }
        timer_fineclustering.Toc();

        // Now we move to the last refinement phase in which we perform clustering with all
        // _n_clusters. Recall our initial buffers for centroids have enough space for _n_clusters.
        if (this->hierarchical_config.verbose) {
            std::cout << "\n=== PHASE 3: REFINEMENT (fine-tuning all " << this->_n_clusters
                      << " clusters) ===" << std::endl;
        }
        this->_n_samples = initial_n_samples;

        // In the refinement phase, we use an even smaller partial d (around 8% of d) because the clusters 
        // are already well-formed, and pruning rate is expected to be high.
        this->_partial_d = std::max<uint32_t>(MIN_PARTIAL_D, this->_vertical_d / 3);

        // We just transfer the state of centroids to the proper class variables, no rotation.
        auto final_refinement_pdx_wrapper =
            SetupCentroids(final_centroids.data(), this->_n_clusters);

        // (RunIteration with is_first_iter=false will swap _horizontal_centroids and
        // _prev_centroids) Copy final_centroids to _prev_centroids so the swap in RunIteration
        // works correctly We could avoid this copies by managing an offset on assignments and
        // centroids in the core functions of SuperKMeans. But this would just complicate the code.
        {
            SKM_PROFILE_SCOPE("copy_final_centroids_and_assignments");
            memcpy(
                static_cast<void*>(this->_prev_centroids.data()),
                static_cast<void*>(final_centroids.data()),
                sizeof(centroid_value_t) * this->_n_clusters * this->_d
            );
            memcpy(
                static_cast<void*>(this->_assignments.data()),
                static_cast<void*>(final_assignments.data()),
                sizeof(uint32_t) * this->_n_samples
            );
        }
        this->GetL2NormsRowMajor(
            this->_prev_centroids.data(), this->_n_clusters, this->_centroid_norms.data()
        );

        TicToc timer_refinement;
        timer_refinement.Tic();

        // Refinement iterations GEMM+PRUNING
        size_t refinement_iter_idx = 0;
        if (this->hierarchical_config.iters_refinement > 0) {
            //
            // FULL GEMM on low-dimensional data or too few clusters
            //
            if (this->_d < DIMENSION_THRESHOLD_FOR_PRUNING ||
                this->_n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
                for (; refinement_iter_idx < this->hierarchical_config.iters_refinement;
                     ++refinement_iter_idx) {
                    this->template RunIteration<true>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        final_refinement_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr, // queries
                        0,       // n_queries
                        this->_n_samples,
                        this->_n_clusters,
                        refinement_iter_idx,
                        false, // is_first_iter
                        this->hierarchical_iteration_stats.refinement_iteration_stats
                    );
                }
            } else { // Rest of Iterations with GEMM+PRUNING
                // TODO(@lkuffo, high): The only reason I need to compute the data norms (again)
                //   is because we are using the same this->_data_norms.data() buffer in the
                //   fineclustering, which replaces the norms that I already calculated before and
                //   put in this buffer.
                this->GetPartialL2NormsRowMajor(
                    data_to_cluster, this->_n_samples, this->_data_norms.data(), this->_partial_d
                );
                for (; refinement_iter_idx < this->hierarchical_config.iters_refinement;
                     ++refinement_iter_idx) {
                    this->template RunIteration<false>(
                        data_to_cluster,
                        tmp_distances_buf.data(),
                        final_refinement_pdx_wrapper,
                        centroids_partial_norms,
                        not_pruned_counts,
                        nullptr, // queries
                        0,       // n_queries
                        this->_n_samples,
                        this->_n_clusters,
                        refinement_iter_idx,
                        false, // is_first_iter
                        this->hierarchical_iteration_stats.refinement_iteration_stats
                    );
                }
            }
        }
        timer_refinement.Toc();

        if (this->hierarchical_config.verbose) {
            std::cout << "Mesoclustering time: " << timer_mesoclustering.GetMilliseconds() << " ms"
                      << std::endl;
            std::cout << "Fineclustering time: " << timer_fineclustering.GetMilliseconds() << " ms"
                      << std::endl;
            std::cout << "Refinement time: " << timer_refinement.GetMilliseconds() << " ms"
                      << std::endl;
            std::cout << "Total time: "
                      << timer_mesoclustering.GetMilliseconds() +
                             timer_fineclustering.GetMilliseconds() +
                             timer_refinement.GetMilliseconds()
                      << " ms" << std::endl;
        }
        this->_trained = true;

        // TODO(@lkuffo, high): If unrotate_centroids is false, this computes incorrect assignments
        //   because it's using unrotated data with rotated output_centroids
        auto output_centroids =
            this->GetOutputCentroids(this->hierarchical_config.unrotate_centroids);
        if (this->hierarchical_config.perform_assignments) {
            // TODO(@lkuffo, high: Need a fast assign, for now we use the iter_idx-1 assignments)
            this->_assignments = this->Assign(data, output_centroids.data(), n, this->_n_clusters);
        }
        if (this->hierarchical_config.verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }

    /**
     * @brief Calculate the number of mesoclusters for a given number of clusters
     *
     * @param n_clusters Total number of clusters
     * @return Number of mesoclusters
     */
    static size_t GetNMesoclusters(const size_t n_clusters) {
        return static_cast<size_t>(std::round(std::sqrt(n_clusters)));
    }

    /**
     * @brief Computes the number of vectors to sample based on sampling_fraction.
     *
     * @param n Total number of vectors
     * @return Number of vectors to sample
     */
    [[nodiscard]] size_t GetNVectorsToSample(const size_t n, size_t n_clusters) const override {
        if (this->hierarchical_config.sampling_fraction == 1.0) {
            return n;
        }
        auto samples_by_n =
            static_cast<size_t>(std::floor(n * this->hierarchical_config.sampling_fraction));
        return samples_by_n;
    }

    /**
     * @brief Override SplitClusters with more aggressive balancing similar to cuVS
     *
     * This version not only handles empty clusters but also actively rebalances
     * small clusters (those below a threshold) by moving their centers toward
     * points from larger clusters.
     *
     * @param n_samples Total number of samples
     * @param n_clusters Number of clusters
     */
    void SplitClusters(const size_t n_samples, const size_t n_clusters) override {
        constexpr float CENTER_ADJUSTMENT_WEIGHT =
            7.0f; // Weight for current center in weighted average
        constexpr float BALANCING_THRESHOLD =
            0.25f; // Clusters smaller than 25% of average are adjusted

        this->_n_split = 0;
        std::mt19937 rng(this->_config.seed);
        auto _horizontal_centroids_p = this->_horizontal_centroids.data();

        size_t average_size = n_samples / n_clusters;
        size_t threshold_size = static_cast<size_t>(average_size * BALANCING_THRESHOLD);
        {
            SKM_PROFILE_SCOPE("consolidate/empty");
            for (size_t ci = 0; ci < n_clusters; ci++) {
                if (this->_cluster_sizes[ci] == 0) {
                    size_t cj;
                    for (cj = 0; true; cj = (cj + 1) % n_clusters) {
                        float p = (this->_cluster_sizes[cj] - 1.0f) /
                                static_cast<float>(n_samples - n_clusters);
                        float r = std::uniform_real_distribution<float>(0, 1)(rng);
                        if (r < p) {
                            break;
                        }
                    }

                    memcpy(
                        (void*) (_horizontal_centroids_p + ci * this->_d),
                        (void*) (_horizontal_centroids_p + cj * this->_d),
                        sizeof(centroid_value_t) * this->_d
                    );

                    // Small symmetric perturbation
                    for (size_t j = 0; j < this->_d; j++) {
                        if (j % 2 == 0) {
                            _horizontal_centroids_p[ci * this->_d + j] *=
                                1.0f + CENTROID_PERTURBATION_EPS;
                            _horizontal_centroids_p[cj * this->_d + j] *=
                                1.0f - CENTROID_PERTURBATION_EPS;
                        } else {
                            _horizontal_centroids_p[ci * this->_d + j] *=
                                1.0f - CENTROID_PERTURBATION_EPS;
                            _horizontal_centroids_p[cj * this->_d + j] *=
                                1.0f + CENTROID_PERTURBATION_EPS;
                        }
                    }

                    // Assume even split of the cluster
                    this->_cluster_sizes[ci] = this->_cluster_sizes[cj] / 2;
                    this->_cluster_sizes[cj] -= this->_cluster_sizes[ci];
                    this->_n_split++;
                }
            }
        }

        std::cout << "n_split before smaller clusters balancing: " << this->_n_split << std::endl;

        // Adjust small clusters (cuVS-style balancing)
        // Pick large clusters with probability proportional to their size
        {
            SKM_PROFILE_SCOPE("consolidate/balancing");
            for (size_t ci = 0; ci < n_clusters; ci++) {
                size_t csize = this->_cluster_sizes[ci];
                if (csize == 0 || csize > threshold_size)
                    continue;

                // Find a large cluster with probability proportional to its size
                size_t large_cluster_idx;
                for (large_cluster_idx = 0; true;
                    large_cluster_idx = (large_cluster_idx + 1) % n_clusters) {
                    size_t large_size = this->_cluster_sizes[large_cluster_idx];
                    if (large_size < average_size)
                        continue;
                    // Probability proportional to how much larger this cluster is than average
                    float p = static_cast<float>(large_size - average_size + 1) /
                            static_cast<float>(n_samples - average_size * n_clusters + n_clusters);
                    float r = std::uniform_real_distribution<float>(0, 1)(rng);
                    if (r < p) {
                        break; // Found our cluster to be split
                    }
                }

                // Adjust the center of the selected smaller cluster to gravitate towards
                // a sample from the selected larger cluster.
                // Weight of the current center for the weighted average.
                // We dump it for anomalously small clusters, but keep constant otherwise.
                float wc = std::min(static_cast<float>(csize), CENTER_ADJUSTMENT_WEIGHT);
                float wd = 1.0f; // Weight for the datapoint used to shift the center.
                for (size_t j = 0; j < this->_d; j++) {
                    float val = 0.0f;
                    val += wc * _horizontal_centroids_p[ci * this->_d + j];
                    val += wd * _horizontal_centroids_p[large_cluster_idx * this->_d + j];
                    val /= (wc + wd);
                    _horizontal_centroids_p[ci * this->_d + j] = val;
                }

                this->_n_split++;
            }
        }
    }

    /*
     * Compact data assigned to a mesocluster in mesocluster_buffer using precomputed indices
     * Data is already rotated, so we just have to copy
     * Additionally, we have to copy their norms in a sequential buffer to not recompute them
     */
    void CompactMesoclusterToBuffer(
        const size_t mesocluster_size,
        const vector_value_t* SKM_RESTRICT data,
        vector_value_t* SKM_RESTRICT mesocluster_buffer,
        uint32_t* SKM_RESTRICT assignments_indirection_buffer,
        const size_t* SKM_RESTRICT mesocluster_indices
    ) {
        SKM_PROFILE_SCOPE("compact_mesocluster");
#pragma omp parallel for if (this->_n_threads > 1) num_threads(this->_n_threads)
        for (size_t j = 0; j < mesocluster_size; ++j) {
            size_t i = mesocluster_indices[j];
            this->_data_norms[j] = immutable_data_norms[i];
            assignments_indirection_buffer[j] = i;
            memcpy(
                static_cast<void*>(mesocluster_buffer + j * this->_d),
                static_cast<const void*>(data + i * this->_d),
                sizeof(vector_value_t) * this->_d
            );
        }
    }

    /**
     * @brief Arrange fine clusters proportionally to mesocluster sizes
     *
     * Allocates the total number of clusters across mesoclusters proportionally
     * to their sizes, ensuring balanced distribution.
     *
     * @param n_clusters Total number of fine clusters to distribute
     * @param n_mesoclusters Number of mesoclusters
     * @param n_samples Total number of samples
     * @param mesocluster_sizes Sizes of each mesocluster
     * @return Vector of fine cluster counts per mesocluster
     */
    std::vector<size_t> ArrangeFineClusters(
        size_t n_clusters,
        size_t n_mesoclusters,
        size_t n_samples,
        const std::vector<uint32_t>& mesocluster_sizes
    ) {
        SKM_PROFILE_SCOPE("arrange_fine_clusters");
        std::vector<size_t> fine_clusters_nums(n_mesoclusters);

        size_t n_clusters_remaining = n_clusters;
        size_t n_nonempty_mesoclusters_remaining = 0;
        for (size_t i = 0; i < n_mesoclusters; ++i) {
            if (mesocluster_sizes[i] > 0) {
                n_nonempty_mesoclusters_remaining++;
            }
        }

        size_t n_samples_remaining = n_samples;
        for (size_t i = 0; i < n_mesoclusters; ++i) {
            if (i < n_mesoclusters - 1) {
                // Handle empty mesoclusters
                if (mesocluster_sizes[i] == 0) {
                    fine_clusters_nums[i] = 0;
                } else {
                    n_nonempty_mesoclusters_remaining--;
                    double proportion =
                        static_cast<double>(n_clusters_remaining * mesocluster_sizes[i]) /
                        static_cast<double>(n_samples_remaining);
                    size_t allocated = static_cast<size_t>(proportion + 0.5);
                    allocated = std::min(
                        allocated, n_clusters_remaining - n_nonempty_mesoclusters_remaining
                    );
                    fine_clusters_nums[i] = std::max(allocated, size_t{1});
                }
            } else {
                // Last mesocluster gets all remaining clusters
                fine_clusters_nums[i] = n_clusters_remaining;
            }
            n_clusters_remaining -= fine_clusters_nums[i];
            n_samples_remaining -= mesocluster_sizes[i];
        }

        return fine_clusters_nums;
    }

    void GetTrueAssignmentsFromIndirectionBuffer(
        const uint32_t* SKM_RESTRICT assignments_indirection_buffer,
        uint32_t* SKM_RESTRICT output_assignments,
        const uint32_t* SKM_RESTRICT input_assignments,
        const size_t n_samples_in_mesocluster,
        const size_t cluster_id_offset
    ) {
        SKM_PROFILE_SCOPE("get_indirection_assignments");
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
     * @brief Setup centroids to be used for the refinement clustering phase
     *
     * @param centroids Centroids to setup
     * @param n_clusters Number of centroids to setupt
     * @return PDXLayout wrapper for the centroids
     */
    PDXLayout<q, alpha> SetupCentroids(
        const centroid_value_t* SKM_RESTRICT centroids,
        const size_t n_clusters
    ) {
        SKM_PROFILE_SCOPE("consolidate");
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
    std::vector<centroid_value_t> final_centroids;
    HierarchicalSuperKMeansConfig hierarchical_config;
    HierarchicalSuperKMeansIterationStats hierarchical_iteration_stats;
};

} // namespace skmeans
