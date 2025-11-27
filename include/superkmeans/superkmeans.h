#pragma once

#include <Eigen/Eigen/Dense>
#include <iomanip>
#include <omp.h>
#include <random>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.hpp"
#include "superkmeans/distance_computers/batch_computers.hpp"
#include "superkmeans/pdx/pdxearch.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/profiler.hpp"

namespace skmeans {
template <Quantization q = f32, DistanceFunction alpha = l2, class Pruner = ADSamplingPruner<q>>
class SuperKMeans {
    using centroid_value_t = skmeans_centroid_value_t<q>;
    using vector_value_t = skmeans_value_t<q>;
    using layout_t = PDXLayout<q, alpha, Pruner>;
    using knn_candidate_t = KNNCandidate<q>;
    using distance_t = skmeans_distance_t<q>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixC = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using VectorR = Eigen::VectorXf;
    using batch_computer = BatchComputer<alpha, q>;

  public:
    SuperKMeans(
        size_t n_clusters,
        size_t dimensionality,
        uint32_t iters = 25,
        float sampling_fraction = 0.50,
        bool verbose = false,
        uint32_t n_threads = 1,
        float tol = 1e-5
    )
        : _iters(iters), _n_clusters(n_clusters), _sampling_fraction(sampling_fraction),
          _d(dimensionality), _trained(false), verbose(verbose), N_THREADS(n_threads), tol(tol) {
        SKMEANS_ENSURE_POSITIVE(n_clusters);
        SKMEANS_ENSURE_POSITIVE(iters);
        SKMEANS_ENSURE_POSITIVE(sampling_fraction);
        SKMEANS_ENSURE_POSITIVE(dimensionality);
        if (sampling_fraction > 1.0) {
            throw std::invalid_argument("sampling_fraction must be smaller than 1");
        }
        // Set global thread count
        g_n_threads = n_threads;
        _pruner = std::make_unique<Pruner>(dimensionality, PRUNER_INITIAL_THRESHOLD);
    }

    /**
     * @brief Performs the clustering on the provided data.
     *
     * @param data The data matrix
     * @param n Number of points (rows) in the data matrix
     * @param d Number of dimensions (cols) in the data matrix
     * @param verbose Whether to use verbose output. Defaults to false.
     * @param num_threads Number of CPU threads to use (set to -1 to use all cores)
     * @param ann_explore_fraction Fraction of centroids to explore for recall computation (0.0 to 1.0). Default 0.01 (1%).
     * @param unrotate_centroids Whether to unrotate centroids before returning (default true). Set to false to get rotated centroids.
     * @return std::vector<skmeans_centroid_value_t<q>> Trained centroids
     */
    std::vector<skmeans_centroid_value_t<q>> Train(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        const vector_value_t* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0,
        const bool sample_queries = false,
        const size_t objective_k = 100,
        const float ann_explore_fraction = 0.01f,
        const bool unrotate_centroids = true
    ) {
        SKMEANS_ENSURE_POSITIVE(n);
        if (_trained) {
            throw std::runtime_error("The clustering has already been trained");
        }

        if (n < _n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters"
            );
        }
        if (n_queries > 0 && queries == nullptr && !sample_queries) {
            throw std::invalid_argument("Queries must be provided if n_queries > 0 and sample_queries is false");
        }
        const vector_value_t* SKM_RESTRICT data_p = data;
        _n_samples = GetNVectorsToSample(n);

        {
            SKM_PROFILE_SCOPE("allocator");
            _centroids.resize(_n_clusters * _d);
            _horizontal_centroids.resize(_n_clusters * _d);
            _prev_centroids.resize(_n_clusters * _d);
            _cluster_sizes.resize(_n_clusters);
            _reciprocal_cluster_sizes.resize(_n_clusters);
            _assignments.resize(n);
            _distances.resize(n);

            data_norms.resize(_n_samples);
            centroid_norms.resize(_n_clusters);
        }
        std::vector<distance_t> all_distances(X_BATCH_SIZE * Y_BATCH_SIZE);

        // TODO(@lkuffo, med): If metric is dp, normalize the vectors so we can use l2

        if (verbose) {
            std::cout << "Generating centroids..." << std::endl;
        }
        //! _centroids and _partial_horizontal_centroids are always wrapped with the PDXLayout object
        _vertical_d = PDXLayout<q, alpha, Pruner>::GetDimensionSplit(_d).vertical_d;
        // Set initial_partial_d dynamically as half of vertical_d
        _initial_partial_d = std::max<uint32_t>(8, _vertical_d / 2);
        // Ensure initial_partial_d doesn't exceed vertical_d to avoid double-counting dimensions
        // when BLAS computes more dimensions than the vertical block contains
        if (_initial_partial_d > _vertical_d) {
            _initial_partial_d = _vertical_d;
        }
        std::cout << "Vertical D: " << _vertical_d << std::endl;
        std::cout << "Horizontal D: " << _d - _vertical_d << std::endl;
        std::cout << "Initial Partial D: " << _initial_partial_d << std::endl;
        {
            SKM_PROFILE_SCOPE("allocator");
            _partial_horizontal_centroids.resize(_n_clusters * _vertical_d);
        }
        auto centroids_pdx_wrapper = GenerateCentroids(data_p, n);
        if (verbose) {
            std::cout << "Sampling data..." << std::endl;
        }

        std::vector<vector_value_t> data_samples_buffer;
        SampleVectors(data_p, data_samples_buffer, n, _n_samples);
        auto data_to_cluster = data_samples_buffer.data();

        // TODO(@lkuffo, low): I don't like this rotated_initial_centroids variable
        std::vector<centroid_value_t> rotated_initial_centroids(_n_clusters * _d);
        {
            SKM_PROFILE_SCOPE("rotator");
            std::cout << "Rotating..." << std::endl;
            _pruner->Rotate(_horizontal_centroids.data(), rotated_initial_centroids.data(), _n_clusters);
        }

        // Getting norms for data and initial centroids (full norms)
        GetL2NormsRowMajor(data_to_cluster, _n_samples, data_norms.data());
        GetL2NormsRowMajor(rotated_initial_centroids.data(), _n_clusters, centroid_norms.data());

        std::vector<vector_value_t> rotated_queries;
        if (n_queries) {
            // Compute number of centroids to explore from the fraction parameter (minimum 1)
            _centroids_to_explore = std::max<size_t>(static_cast<size_t>(_n_clusters * ann_explore_fraction), 1);
            std::cout << " -----> Centroids to explore: " << _centroids_to_explore 
                      << " (" << ann_explore_fraction * 100.0f << "% of " << _n_clusters << ")" << std::endl;
            {
                SKM_PROFILE_SCOPE("allocator");
                _gt_assignments.resize(n_queries * objective_k);
                _gt_distances.resize(n_queries * objective_k);
            }
            // Create temporary buffer for rotated queries and rotate them
            rotated_queries.resize(n_queries * _d);
            if (sample_queries) {
                std::cout << "Sampling queries from data..." << std::endl;
                SampleVectors<false>(data_to_cluster, rotated_queries, _n_samples, n_queries);
            } else {
                // We already did a validation step to ensure that queries is not nullptr
                SKM_PROFILE_SCOPE("rotator");
                _pruner->Rotate(queries, rotated_queries.data(), n_queries);
            }
            // Compute and cache query norms once (used by ComputeRecall in each iteration)
            _query_norms.resize(n_queries);
            GetL2NormsRowMajor(rotated_queries.data(), n_queries, _query_norms.data());
            GetGTAssignmentsAndDistances(data_to_cluster, _n_samples, rotated_queries.data(), n_queries, objective_k);

            // Print the assignments and distances of the first 10 queries
            // if (verbose) {
            //     std::cout << "First 10 GT Assignments and Distances:" << std::endl;
            //     for (size_t i = 0; i < 100; ++i) {
            //         std::cout << "Query " << i << ":" << std::endl;
            //         for (size_t j = 0; j < objective_k; ++j) {
            //             std::cout << "  Assignment: " << _gt_assignments[i * objective_k + j] << " Distance: " << _gt_distances[i * objective_k + j] << std::endl;
            //         }
            //     }
            //     std::cout << std::endl;
            // }
        }

        // First iteration: Only Blas
        InitAssignAndUpdateCentroids(
            data_to_cluster,
            rotated_initial_centroids.data(),
            data_norms.data(),
            centroid_norms.data(),
            all_distances.data(),
            _n_samples
        );
        ConsolidateCentroids();
        ComputeCost();
        ComputeShift();
        if (alpha == dp) {
            PostprocessCentroids();
        }
        if (n_queries) {
            _recall = ComputeRecall(rotated_queries.data(), n_queries, objective_k, _centroids_to_explore);
        }
        size_t iter_idx = 1;
        if (verbose)
            std::cout << "Iteration 1" << "/" << _iters << " | Objective: " << cost
                      << " | Shift: " << shift << " | Split: " << _n_split
                      << " | Recall: " << _recall << std::endl << std::endl;
        // End of First iteration

        if (_iters <= 1) {
            auto output_centroids = GetOutputCentroids(unrotate_centroids);
            Profiler::Get().PrintHierarchical();
            return output_centroids;
        }

        // Special path for low-dimensional data: use BLAS-only for all iterations
        if (_d < 128) {
            for (; iter_idx < _iters; ++iter_idx) {
                // Save current centroids for shift computation
                std::copy(_horizontal_centroids.begin(), _horizontal_centroids.end(), _prev_centroids.begin());
                // Recompute centroid norms for the updated centroids
                GetL2NormsRowMajor(_horizontal_centroids.data(), _n_clusters, centroid_norms.data());
                InitAssignAndUpdateCentroids(
                    data_to_cluster,
                    _horizontal_centroids.data(),
                    data_norms.data(),
                    centroid_norms.data(),
                    all_distances.data(),
                    _n_samples
                );
                ConsolidateCentroids();
                ComputeCost();
                ComputeShift();
                if (alpha == dp) {
                    PostprocessCentroids();
                }
                if (n_queries) {
                    // Update centroid norms to match the NEW centroids (after ConsolidateCentroids)
                    GetL2NormsRowMajor(_horizontal_centroids.data(), _n_clusters, centroid_norms.data());
                    _recall = ComputeRecall(rotated_queries.data(), n_queries, objective_k, _centroids_to_explore);
                }
                if (verbose)
                    std::cout << "Iteration " << iter_idx + 1 << "/" << _iters
                              << " | Objective: " << cost << " | Shift: " << shift
                              << " | Split: " << _n_split
                              << " | Recall: " << _recall << " [BLAS-only]" << std::endl << std::endl;
            }
            _trained = true;
            auto output_centroids = GetOutputCentroids(unrotate_centroids);
            if (verbose) {
                Profiler::Get().PrintHierarchical();
            }
            return output_centroids;
        }


        // Rest of iterations
        std::vector<vector_value_t> centroids_partial_norms(_n_clusters);
        // Buffer to store per-vector not-pruned counts for tuning _initial_partial_d
        std::vector<size_t> not_pruned_counts(_n_samples);
        GetPartialL2NormsRowMajor(data_to_cluster, _n_samples, data_norms.data());
        for (; iter_idx < _iters; ++iter_idx) {
            // Save current centroids for shift computation
            std::copy(_horizontal_centroids.begin(), _horizontal_centroids.end(), _prev_centroids.begin());
            GetL2NormsRowMajor(
                _horizontal_centroids.data(), _n_clusters, centroids_partial_norms.data(), _initial_partial_d
            );
            // Reset the not-pruned counts buffer
            std::fill(not_pruned_counts.begin(), not_pruned_counts.end(), 0);
            AssignAndUpdateCentroidsPartialBatched(
                data_to_cluster,
                _horizontal_centroids.data(),
                data_norms.data(),
                centroids_partial_norms.data(),
                all_distances.data(),
                centroids_pdx_wrapper,
                _n_samples,
                not_pruned_counts.data()
            );
            
            // Tune _initial_partial_d based on the average not-pruned percentage
            bool partial_d_changed = false;
            float avg_not_pruned_pct = TuneInitialPartialD(not_pruned_counts.data(), _n_samples, _n_clusters, partial_d_changed);
            
            // If _initial_partial_d changed, recompute the data norms with the new partial_d
            if (partial_d_changed) {
                GetPartialL2NormsRowMajor(data_to_cluster, _n_samples, data_norms.data());
            }
            
            ConsolidateCentroids();
            ComputeCost();
            ComputeShift();
            if (alpha == dp) {
                PostprocessCentroids();
            }
            if (n_queries) {
                // Update centroid norms with FULL norms for recall computation
                // (PDX uses partial norms for distance computation, but recall needs full norms)
                GetL2NormsRowMajor(_horizontal_centroids.data(), _n_clusters, centroid_norms.data());
                _recall = ComputeRecall(rotated_queries.data(), n_queries, objective_k, _centroids_to_explore);
            }
            if (verbose)
                std::cout << "Iteration " << iter_idx + 1 << "/" << _iters
                          << " | Objective: " << cost << " | Shift: " << shift
                          << " | Split: " << _n_split 
                          << " | Recall: " << _recall
                          << " | Not Pruned %: " << avg_not_pruned_pct * 100.0f
                          << " | Partial D: " << _initial_partial_d << std::endl << std::endl;
        }
        //! I only need assignments if sampling_faction < 1
        if (_sampling_fraction < 1.0f) {
            // TODO(@lkuffo, supercrit): If data is sampled, then assignments are only done to the first n_samples vectors (remember 
            // that we sampled the first n_samples vectors from the original data). To do this we need to do a finall call to the only-blas batch computer
            // only for these remaining points.
            // Assign(data, centroids_pdx_wrapper, n);
        }

        _trained = true;
        auto output_centroids = GetOutputCentroids(unrotate_centroids);
        if (verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }


    /**
     * @brief Assign given data points to their nearest cluster.
     *
     * NOTE: The dimensionality of the data should match the dimensionality of the
     * data that the clustering was trained on.
     *
     * @param data The data matrix
     * @param n The number of data points (rows) in the data matrix
     * @return std::vector<uint32_t> Clustering assignments as a vector of
     * sequential ids of the points assigned to the corresponding cluster
     */
    void Assign(
        const vector_value_t* SKM_RESTRICT data,
        const layout_t& pdx_centroids,
        const size_t n
    ) {
        auto data_p = data;
        // TODO(@lkuffo, med): Skip the vectors that were used as samples
        for (size_t i = 0; i < n; ++i) {
            // PDXearch per vector
            std::vector<knn_candidate_t> assignment = pdx_centroids.searcher->Search(data_p, 1);
            auto assignment_idx = assignment[0].index;
            _assignments[i] = assignment_idx;
            data_p += _d;
        }
    }

    /**
     * @brief Assign given data points to their nearest cluster.
     *
     * NOTE: The dimensionality of the data should match the dimensionality of the
     * data that the clustering was trained on.
     *
     * @param data The data matrix
     * @param n The number of data points (rows) in the data matrix
     * @return std::vector<uint32_t> Clustering assignments as a vector of
     * sequential ids of the points assigned to the corresponding cluster
     */
    void InitAssignAndUpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT rotated_initial_centroids,
        const vector_value_t* SKM_RESTRICT data_norms,
        const vector_value_t* SKM_RESTRICT centroid_norms,
        distance_t* SKM_RESTRICT all_distances,
        const size_t n
    ) {
        batch_computer::Batched_XRowMajor_YRowMajor(
            data,
            rotated_initial_centroids,
            n,
            _n_clusters,
            _d,
            data_norms,
            centroid_norms,
            _assignments.data(),
            _distances.data(),
            all_distances
        );
        std::fill(_horizontal_centroids.begin(), _horizontal_centroids.end(), 0.0);
        std::fill(_cluster_sizes.begin(), _cluster_sizes.end(), 0);
        cost = 0.0;
        UpdateCentroids(data, n);
    }

    void AssignAndUpdateCentroidsPartialBatched(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT partial_rotated_centroids,
        const vector_value_t* SKM_RESTRICT partial_data_norms,
        const vector_value_t* SKM_RESTRICT partial_centroid_norms,
        distance_t* SKM_RESTRICT all_distances,
        const layout_t& pdx_centroids,
        const size_t n,
        size_t* out_not_pruned_counts = nullptr
    ) {
        cost = 0.0;
        batch_computer::Batched_XRowMajor_YRowMajor_PartialD(
            data,
            partial_rotated_centroids,
            n,
            _n_clusters,
            _d,
            partial_data_norms,
            partial_centroid_norms,
            _assignments.data(),
            _distances.data(),
            all_distances,
            pdx_centroids,
            _initial_partial_d,
            out_not_pruned_counts
        );
        std::fill(_horizontal_centroids.begin(), _horizontal_centroids.end(), 0.0);
        std::fill(_cluster_sizes.begin(), _cluster_sizes.end(), 0);
        UpdateCentroids(data, n);
    }

    void UpdateCentroids(const vector_value_t* SKM_RESTRICT data, const size_t n) {
        SKM_PROFILE_SCOPE("update_centroids");
#pragma omp parallel num_threads(g_n_threads)
        {
            uint32_t nt = g_n_threads;
            uint32_t rank = omp_get_thread_num();
            // This thread is taking care of centroids c0:c1
            size_t c0 = (_n_clusters * rank) / nt;
            size_t c1 = (_n_clusters * (rank + 1)) / nt;
            for (size_t i = 0; i < n; i++) {
                int64_t ci = _assignments[i];
                assert(ci >= 0 && ci < _n_clusters);
                if (ci >= c0 && ci < c1) {
                    auto vector_p = data + i * _d;
                    _cluster_sizes[ci] += 1;
                    UpdateCentroid(vector_p, ci);
                }
            }
        }
    }

    SKM_ALWAYS_INLINE void UpdateCentroid(
        const vector_value_t* SKM_RESTRICT vector,
        const uint32_t cluster_idx
    ) {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < _d; ++i) {
            _horizontal_centroids[cluster_idx * _d + i] += vector[i];
        }
    }

    void SplitClusters() {
        _n_split = 0;
        std::default_random_engine rng;
        auto _horizontal_centroids_p = _horizontal_centroids.data();
        for (size_t ci = 0; ci < _n_clusters; ci++) {
            if (_cluster_sizes[ci] == 0) { // Need to redefine a centroid
                size_t cj;
                for (cj = 0; true; cj = (cj + 1) % _n_clusters) {
                    // Probability to pick this cluster for split
                    float p = (_cluster_sizes[cj] - 1.0) / (float) (_n_samples - _n_clusters);
                    float r = std::uniform_real_distribution<float>(0, 1)(rng);
                    if (r < p) {
                        break; // Found our cluster to be split
                    }
                }
                memcpy(
                    (void*) (_horizontal_centroids_p + ci * _d),
                    (void*) (_horizontal_centroids_p + cj * _d),
                    sizeof(centroid_value_t) * _d
                );

                // Small symmetric perturbation
                for (size_t j = 0; j < _d; j++) {
                    if (j % 2 == 0) {
                        _horizontal_centroids_p[ci * _d + j] *= 1 + CENTROID_PERTURBATION_EPS;
                        _horizontal_centroids_p[cj * _d + j] *= 1 - CENTROID_PERTURBATION_EPS;
                    } else {
                        _horizontal_centroids_p[ci * _d + j] *= 1 - CENTROID_PERTURBATION_EPS;
                        _horizontal_centroids_p[cj * _d + j] *= 1 + CENTROID_PERTURBATION_EPS;
                    }
                }

                // Assume even split of the cluster
                _cluster_sizes[ci] = _cluster_sizes[cj] / 2;
                _cluster_sizes[cj] -= _cluster_sizes[ci];
                _n_split++;
            }
        }
    }

    void ConsolidateCentroids() {
        SKM_PROFILE_SCOPE("consolidate");
        {
            SKM_PROFILE_SCOPE("consolidate/splitting");
#pragma omp parallel for if (g_n_threads > 1) num_threads(g_n_threads)
            for (size_t i = 0; i < _n_clusters; ++i) {
                auto _horizontal_centroids_p = _horizontal_centroids.data() + i * _d;
                if (_cluster_sizes[i] == 0) {
                    continue;
                }
                float mult_factor = 1.0 / _cluster_sizes[i];
#pragma clang loop vectorize(enable)
                for (size_t j = 0; j < _d; ++j) {
                    _horizontal_centroids_p[j] *= mult_factor;
                }
            }
            SplitClusters();
        }
        {
            SKM_PROFILE_SCOPE("consolidate/pdxify");
            //! This updates the object within the pdx_layout wrapper
            PDXLayout<q, alpha, Pruner>::template PDXify<false>(
                _horizontal_centroids.data(), _centroids.data(), _n_clusters, _d
            );
            CentroidsToAuxiliaryHorizontal();
        }
    }

    void ComputeCost() {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < _n_samples; ++i) {
            cost += _distances[i];
        }
    }

    void ComputeShift() {
        SKM_PROFILE_SCOPE("shift");
        Eigen::Map<const MatrixR> new_mat(_horizontal_centroids.data(), _n_clusters, _d);
        Eigen::Map<const MatrixR> prev_mat(_prev_centroids.data(), _n_clusters, _d);
        MatrixR diff = new_mat - prev_mat;
        shift = 0.0f;
#pragma omp parallel for reduction(+:shift) num_threads(g_n_threads)
        for (size_t i = 0; i < _n_clusters; ++i) {
            shift += diff.row(i).squaredNorm();
        }
        shift /= (_n_clusters * _d);
    }

    void GetGTAssignmentsAndDistances(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        const vector_value_t* SKM_RESTRICT queries,
        const size_t n_queries,
        const size_t objective_k
    ) {
        SKM_PROFILE_SCOPE("gt_assignments");
        std::vector<distance_t> tmp_distances_buffer(X_BATCH_SIZE * Y_BATCH_SIZE);
        std::vector<distance_t> query_norms(n_queries);
        GetL2NormsRowMajor(queries, n_queries, query_norms.data());
        batch_computer::Batched_XRowMajor_YRowMajor_TopK(
            queries,
            data,
            n_queries,
            n,
            _d,
            query_norms.data(),
            data_norms.data(),
            objective_k,
            _gt_assignments.data(),
            _gt_distances.data(),
            tmp_distances_buffer.data()
        );
    }

    float ComputeRecall(const vector_value_t* SKM_RESTRICT queries, const size_t n_queries, const size_t objective_k, const size_t centroids_to_explore) {
        SKM_PROFILE_SCOPE("recall");
        std::vector<distance_t> tmp_distances_buffer(X_BATCH_SIZE * Y_BATCH_SIZE);
        std::vector<uint32_t> promising_centroids(n_queries * centroids_to_explore);
        std::vector<distance_t> distances(n_queries * centroids_to_explore);
        batch_computer::Batched_XRowMajor_YRowMajor_TopK(
            queries,
            _horizontal_centroids.data(),
            n_queries,
            _n_clusters,
            _d,
            _query_norms.data(),
            centroid_norms.data(),
            centroids_to_explore,
            promising_centroids.data(),
            distances.data(),
            tmp_distances_buffer.data()
        );
        // For each query, compute recall@objective_k: how many of the GT clusters are found in the top-64 assignments
        // Recall per query = (# matched GT assignments in top-64) / objective_k
        // Final recall = average over all queries
        float sum_recall = 0.0f;
        for (size_t i = 0; i < n_queries; ++i) {
            size_t found_in_query = 0;
            // For each GT assignment for query q
            for (size_t j = 0; j < objective_k; ++j) {
                uint32_t gt = _gt_assignments[i * objective_k + j]; // gt is a vector index
                // Check if this GT assignment is present in the top-64 assignments for this query
                bool found = false;
                for (size_t t = 0; t < centroids_to_explore; ++t) {
                    // If a promising centroid is the same as the GT centroid assignment, then we have a match
                    if (promising_centroids[i * centroids_to_explore + t] == _assignments[gt]) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    ++found_in_query;
                }
            }
            sum_recall += static_cast<float>(found_in_query) / static_cast<float>(objective_k);
        }
        return sum_recall / static_cast<float>(n_queries);
    }
    std::vector<skmeans_centroid_value_t<q>> GetCentroids() const { return _centroids; }

    inline size_t GetNClusters() const { return _n_clusters; }

    inline bool IsTrained() const { return _trained; }

  protected:
    PDXLayout<q, alpha, Pruner> GenerateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n
    ) {
        {
            SKM_PROFILE_SCOPE("sampling");
            auto tmp_centroids_p = _horizontal_centroids.data();
            // First `n` samples similar to FAISS'
            for (size_t i = 0; i < _n_clusters; i += 1) {
                // TODO(@lkuffo, low): What if centroid scalar_t are not the same size of vector ones
                memcpy(
                    (void*) tmp_centroids_p, (void*) (data + (i * _d)), sizeof(centroid_value_t) * _d
                );
                tmp_centroids_p += _d;
            }
        }
        // We populate the _centroids buffer with the centroids in the PDX layout
        std::vector<centroid_value_t> rotated_centroids(_n_clusters * _d);
        {
            SKM_PROFILE_SCOPE("rotator");
            _pruner->Rotate(_horizontal_centroids.data(), rotated_centroids.data(), _n_clusters);
        }
        {
            SKM_PROFILE_SCOPE("consolidate/pdxify");
            PDXLayout<q, alpha, Pruner>::template PDXify<false>(
                rotated_centroids.data(), _centroids.data(), _n_clusters, _d
            );
        }
        //! We wrap _centroids and _partial_horizontal_centroids in the PDXLayout wrapper
        //! Any updates to these objects is reflected in the PDXLayout
        auto pdx_centroids = PDXLayout<q, alpha, Pruner>(
            _centroids.data(), *_pruner, _n_clusters, _d, _partial_horizontal_centroids.data()
        );
        return pdx_centroids;
    }

    void GetPartialL2NormsRowMajor(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        vector_value_t* SKM_RESTRICT out_norm
    ) {
        SKM_PROFILE_SCOPE("norms_calc");
        Eigen::Map<const MatrixR> e_data(data, n, _d);
        Eigen::Map<VectorR> e_norms(out_norm, n);
        e_norms.noalias() = e_data.leftCols(_initial_partial_d).rowwise().squaredNorm();
    }

    void GetL2NormsRowMajor(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        vector_value_t* SKM_RESTRICT out_norm
    ) {
        SKM_PROFILE_SCOPE("norms_calc");
        Eigen::Map<const MatrixR> e_data(data, n, _d);
        Eigen::Map<VectorR> e_norms(out_norm, n);
        e_norms.noalias() = e_data.rowwise().squaredNorm();
    }

    void CentroidsToAuxiliaryHorizontal() {
        Eigen::Map<MatrixR> hor_centroids(_horizontal_centroids.data(), _n_clusters, _d);
        Eigen::Map<MatrixR> out_aux_centroids(_partial_horizontal_centroids.data(), _n_clusters, _vertical_d);
        out_aux_centroids.noalias() = hor_centroids.leftCols(_vertical_d);
    }

    void GetL2NormsRowMajor(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        vector_value_t* SKM_RESTRICT out_norm,
        const size_t partial_d
    ) {
        SKM_PROFILE_SCOPE("norms_calc");
        Eigen::Map<const MatrixR> e_data(data, n, _d);
        Eigen::Map<VectorR> e_norms(out_norm, n);
        e_norms.noalias() = e_data.leftCols(partial_d).rowwise().squaredNorm();
    }

    /**
     * @brief Tune _initial_partial_d based on the average not-pruned percentage.
     * 
     * A safe range for pruning is between 75% - 90% of vectors pruned (i.e., 10% - 25% not pruned).
     * - If avg_not_pruned_pct > 25% (i.e., less than 75% pruned), we reduce _initial_partial_d by 25%
     *   to be more aggressive in pruning
     * - If avg_not_pruned_pct < 10% (i.e., more than 90% pruned), we increase _initial_partial_d by 25%
     *   to be less aggressive
     * - _initial_partial_d is clamped between 8 and _vertical_d
     * 
     * @param not_pruned_counts Buffer containing per-vector not-pruned counts
     * @param n_samples Number of X vectors
     * @param n_y Number of Y vectors (centroids)
     * @param partial_d_changed Output parameter: set to true if _initial_partial_d was changed
     * @return The computed average not-pruned percentage
     */
    float TuneInitialPartialD(const size_t* not_pruned_counts, size_t n_samples, size_t n_y, bool& partial_d_changed) {
        constexpr float MIN_NOT_PRUNED_PCT = 0.03f;  // 3% not pruned = 97% pruned
        constexpr float MAX_NOT_PRUNED_PCT = 0.10f;  // 10% not pruned = 90% pruned
        constexpr float ADJUSTMENT_FACTOR = 0.10f;   // 10% adjustment
        constexpr uint32_t MIN_PARTIAL_D = 8;
        
        // Calculate average not-pruned percentage from the buffer
        float avg_not_pruned_pct = 0.0f;
        for (size_t i = 0; i < n_samples; ++i) {
            // std::cout << "Not pruned count: " << not_pruned_counts[i] << " out of " << n_y << std::endl;
            avg_not_pruned_pct += static_cast<float>(not_pruned_counts[i]);
        }
        avg_not_pruned_pct /= static_cast<float>(n_samples * n_y);
        
        uint32_t old_partial_d = _initial_partial_d;
        
        if (avg_not_pruned_pct > MAX_NOT_PRUNED_PCT) {
            // Too many vectors not pruned (< MAX_NOT_PRUNED_PCT pruned), need more BLAS dimensions
            // Increase _initial_partial_d by ADJUSTMENT_FACTOR
            uint32_t increase = static_cast<uint32_t>(_initial_partial_d * ADJUSTMENT_FACTOR);
            _initial_partial_d = std::min(_initial_partial_d + std::max(increase, 1u), _vertical_d);
        } else if (avg_not_pruned_pct < MIN_NOT_PRUNED_PCT) {
            // Too few vectors not pruned (> MIN_NOT_PRUNED_PCT pruned), can reduce BLAS dimensions
            // Decrease _initial_partial_d by ADJUSTMENT_FACTOR
            uint32_t decrease = static_cast<uint32_t>(_initial_partial_d * ADJUSTMENT_FACTOR);
            _initial_partial_d = std::max(_initial_partial_d - std::max(decrease, 1u), MIN_PARTIAL_D);
        }
        partial_d_changed = (old_partial_d != _initial_partial_d);
        
        // else: within safe range (75% - 90% pruned), no adjustment needed
        if (verbose && partial_d_changed) {
            std::cout << "Tuning _initial_partial_d: " << old_partial_d << " -> " << _initial_partial_d
                      << " (avg not pruned: " << avg_not_pruned_pct * 100.0f << "%)" << std::endl;
        }
        
        return avg_not_pruned_pct;
    }

    size_t GetNVectorsToSample(const size_t n) const {
        if (_sampling_fraction == 1.0) {
            return n;
        }
        return std::floor(n * _sampling_fraction);
    }

    /**
     * @brief Prepare centroids for output (applies any necessary transformations).
     * @param should_unrotate If true, unrotates centroids to original space; if false, returns rotated centroids.
     * @return Centroids ready for output
     */
    std::vector<centroid_value_t> GetOutputCentroids(bool should_unrotate) {
        if (should_unrotate) {
            SKM_PROFILE_SCOPE("unrotator");
            std::vector<centroid_value_t> unrotated(_n_clusters * _d);
            _pruner->Unrotate(_horizontal_centroids.data(), unrotated.data(), _n_clusters);
            return unrotated;
        }
        return _horizontal_centroids;
    }

    // TODO(@lkuffo, low): Centroids are on PDX, I cant do this... but TMP centroids are not!
    void PostprocessCentroids() {
        auto centroids_p = _centroids.data();
        for (size_t i = 0; i < _n_clusters; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < _d; ++j) {
                sum += centroids_p[i * _d + j] * centroids_p[i * _d + j];
            }
            float norm = std::sqrt(sum);
            for (size_t j = 0; j < _d; ++j) {
                centroids_p[i * _d + j] = centroids_p[i * _d + j] / norm;
            }
        }
    }

    template <bool ROTATE = true>
    void SampleVectors(
        const vector_value_t* SKM_RESTRICT data,
        std::vector<vector_value_t>& out_buffer,
        const size_t n,
        const size_t n_samples
    ) {
        out_buffer.resize(n_samples * _d);
        std::cout << "n_samples: " << n_samples << std::endl;

        // Intermediate buffer needed only when both sampling and rotating
        // (we have not yet implemented rotation in-place)
        std::vector<vector_value_t> samples_tmp;
        const vector_value_t* src_data = data;
        
        if (n_samples < n) {
            SKM_PROFILE_SCOPE("sampling");
            if constexpr (ROTATE) {
                // Need intermediate buffer: sample first, then rotate
                samples_tmp.resize(n_samples * _d);
                const auto jumps = static_cast<size_t>(std::floor((1.0 * n) / n_samples));
                for (size_t i = 0; i < n_samples; i++) {
                    size_t src_vector_idx = i * jumps;
                    memcpy(
                        (void*) (samples_tmp.data() + i * _d),
                        (void*) (data + src_vector_idx * _d),
                        sizeof(vector_value_t) * _d
                    );
                }
                src_data = samples_tmp.data();
            } else {
                // No rotation: sample directly into output buffer
                const auto jumps = static_cast<size_t>(std::floor((1.0 * n) / n_samples));
                for (size_t i = 0; i < n_samples; i++) {
                    size_t src_vector_idx = i * jumps;
                    memcpy(
                        (void*) (out_buffer.data() + i * _d),
                        (void*) (data + src_vector_idx * _d),
                        sizeof(vector_value_t) * _d
                    );
                }
                return;  // Done, no rotation needed
            }
        }

        // Rotate or copy into output buffer
        SKM_PROFILE_SCOPE("rotator");
        if constexpr (ROTATE) {
            _pruner->Rotate(src_data, out_buffer.data(), n_samples);
        } else {
            memcpy(
                (void*) out_buffer.data(),
                (void*) src_data,
                sizeof(vector_value_t) * n_samples * _d
            );
        }
    }

    std::unique_ptr<Pruner> _pruner;

    std::vector<centroid_value_t> _centroids;      // Always keeps the PDX centroids
    std::vector<centroid_value_t> _horizontal_centroids;  // Always keeps the horizontal (row-major) centroids
    std::vector<centroid_value_t> _prev_centroids; // Always keeps the previous iteration centroids
    std::vector<centroid_value_t> _partial_horizontal_centroids; // Partial horizontal centroids (first _vertical_d dimensions)
    std::vector<uint32_t> _assignments;
    std::vector<distance_t> _distances;
    std::vector<uint32_t> _cluster_sizes;

    std::vector<uint32_t> _gt_assignments;
    std::vector<distance_t> _gt_distances;
    std::vector<distance_t> _query_norms;  // Cached query norms (computed once per Train call)

    std::vector<vector_value_t> data_norms;
    std::vector<vector_value_t> centroid_norms;
    std::vector<float> _reciprocal_cluster_sizes;

    const uint32_t _iters;
    const size_t _n_clusters;
    const float _sampling_fraction;
    const size_t _d;
    bool _trained;
    bool verbose;
    float cost;
    float shift;
    size_t _n_samples;
    size_t _n_split;
    uint32_t N_THREADS;
    uint32_t _initial_partial_d = DEFAULT_INITIAL_PARTIAL_D;
    uint32_t _vertical_d;
    float tol;
    float _recall = 0.0f;
    size_t _centroids_to_explore = 64;

};
} // namespace skmeans
