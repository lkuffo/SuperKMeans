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
    // TODO(lkuffo, high): If less than 128 dimensions, then do not even attempt to prune.
    //   Just send everything to batched queries
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
     * @return std::vector<skmeans_centroid_value_t<q>> Trained centroids
     */
    std::vector<skmeans_centroid_value_t<q>> Train(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        const vector_value_t* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0,
        const bool sample_queries = false,
        const size_t objective_k = 100
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

        _allocator_time.Tic();

        _centroids.resize(_n_clusters * _d);
        _tmp_centroids.resize(_n_clusters * _d);
        _prev_centroids.resize(_n_clusters * _d);
        _cluster_sizes.resize(_n_clusters);
        _reciprocal_cluster_sizes.resize(_n_clusters);
        _assignments.resize(n);
        _distances.resize(n);

        // Pruning groups logic
        _pruning_groups.resize(n);

        data_norms.resize(_n_samples);
        centroid_norms.resize(_n_clusters);
        std::vector<distance_t> all_distances(X_BATCH_SIZE * Y_BATCH_SIZE);
        _allocator_time.Toc();

        // TODO(@lkuffo, med): If metric is dp, normalize the vectors so we can use l2

        if (verbose) {
            std::cout << "Generating centroids..." << std::endl;
        }
        //! _centroids and _aux_hor_centroids are always wrapped with the PDXLayout object
        _vertical_d = PDXLayout<q, alpha, Pruner>::GetDimensionSplit(_d).vertical_d;
        std::cout << "Vertical D: " << _vertical_d << std::endl;
        std::cout << "Horizontal D: " << _d - _vertical_d << std::endl;
        _allocator_time.Tic();
        _aux_hor_centroids.resize(_n_clusters * _vertical_d);
        _allocator_time.Toc();
        auto centroids_pdx_wrapper = GenerateCentroids(data_p, n);
        if (verbose) {
            std::cout << "Sampling data..." << std::endl;
        }

        // TODO: This is bad because data_to_cluster may have one buffer or the other, objects
        // life span is not clear
        std::vector<vector_value_t> data_samples_buffer;
        auto data_to_cluster = SampleVectors(data_p, data_samples_buffer, n, _n_samples);

        // TODO(@lkuffo, low): I don't like this rotated_initial_centroids variable
        _allocator_time.Tic();
        std::vector<centroid_value_t> rotated_initial_centroids(_n_clusters * _d);
        _allocator_time.Toc();
        _rotator_time.Tic();
        std::cout << "Rotating..." << std::endl;
        _pruner->Rotate(_tmp_centroids.data(), rotated_initial_centroids.data(), _n_clusters);
        _rotator_time.Toc();

        // Getting norms for data and initial centroids (full norms)
        GetL2NormsRowMajor(data_to_cluster, _n_samples, data_norms.data());
        GetL2NormsRowMajor(rotated_initial_centroids.data(), _n_clusters, centroid_norms.data());

        std::vector<vector_value_t> rotated_queries;
        if (n_queries) {
            // TODO(@lkuffo, crit): Set this to a percentage of the number of clusters defined as a parameter of Train()
            _centroids_to_explore = std::max<size_t>(_n_clusters / 100, 1);
            std::cout << " -----> Centroids to explore: " << _centroids_to_explore << std::endl;
            if (verbose) {
                std::cout << "Getting GT Assignments and Distances for " << n_queries << " queries..." << std::endl;
            }
            _allocator_time.Tic();
            _gt_assignments.resize(n_queries * objective_k);
            _gt_distances.resize(n_queries * objective_k);
            _allocator_time.Toc();
            _rotator_time.Tic();
            // Create temporary buffer for rotated queries and rotate them
            rotated_queries.resize(n_queries * _d);
            if (sample_queries) {
                std::cout << "Sampling queries from data..." << std::endl;
                _sampling_time.Tic();
                SampleVectors<false>(data_to_cluster, rotated_queries, _n_samples, n_queries);
                _sampling_time.Toc();
            } else {
                // We already did a validation step to ensure that queries is not nullptr
                _rotator_time.Tic();
                _pruner->Rotate(queries, rotated_queries.data(), n_queries);
                _rotator_time.Toc();
            }

            _rotator_time.Toc();
            _gt_assignments_time.Tic();
            GetGTAssignmentsAndDistances(data_to_cluster, _n_samples, rotated_queries.data(), n_queries, objective_k);
            _gt_assignments_time.Toc();
            std::cout << "TOTAL GT ASSIGNMENTS TIME " << _gt_assignments_time.accum_time / 1000000000.0 << std::endl;

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
            _recall_time.Tic();
            float cur_recall = ComputeRecall(rotated_queries.data(), n_queries, objective_k, _centroids_to_explore);
            _recall_time.Toc();
            _recall = cur_recall;
        }
        size_t iter_idx = 1;
        if (verbose)
            std::cout << "Iteration 1" << "/" << _iters << " | Objective: " << cost
                      << " | Shift: " << shift << " | Split: " << _n_split
                      << " | Recall: " << _recall << std::endl << std::endl;
        // End of First iteration

        if (_iters <= 1) {
            PrintTimes();
            return _centroids;
        }
        // Second iteration: PDXearch to determine dimensions groupings
        _pruning_groups.clear();
        _pruning_groups_partial_d.clear();
        if (_n_pruning_groups > 1) {
            _pruning_groups_ends.push_back(n);
            _pruning_groups_partial_d.push_back(_initial_partial_d);
            GetGroupsL2NormsRowMajor(data_to_cluster, _n_samples, data_norms.data());
            GetL2NormsRowMajor(
                _tmp_centroids.data(), _n_clusters, centroid_norms.data(), _initial_partial_d
            );
            AssignAndUpdateCentroidsPartialBatched<true>( // Record pruning_group
                data_to_cluster,
                _tmp_centroids.data(),
                data_norms.data(),
                centroid_norms.data(),
                all_distances.data(),
                centroids_pdx_wrapper,
                _n_samples
            );
            ConsolidateCentroids();
            ComputeCost();
            ComputeShift();
            if (alpha == dp) {
                PostprocessCentroids();
            }
            iter_idx += 1;
            if (verbose)
                std::cout << "Iteration 2" << "/" << _iters << " | Objective: " << cost
                          << " | Shift: " << shift << " | Split: " << _n_split 
                          << " | Recall: " << _recall << std::endl << std::endl;
            // End of second iteration
            CreatePruningGroups(data_to_cluster, n);
        } else {
            _pruning_groups_ends.push_back(n);
            _pruning_groups_partial_d.push_back(_initial_partial_d);
        }

        // Rest of iterations
        // We need as many buffers with norms as groups
        _allocator_time.Tic();
        std::vector<vector_value_t> centroid_partial_norms(
            _n_clusters * _pruning_groups_partial_d.size()
        );
        _allocator_time.Toc();
        GetGroupsL2NormsRowMajor(data_to_cluster, _n_samples, data_norms.data());
        for (; iter_idx < _iters; ++iter_idx) {
            GetL2NormsRowMajorPerGroup(
                _tmp_centroids.data(), _n_clusters, centroid_partial_norms.data()
            );
            AssignAndUpdateCentroidsPartialBatched(
                data_to_cluster,
                _tmp_centroids.data(),
                data_norms.data(),
                centroid_partial_norms.data(),
                all_distances.data(),
                centroids_pdx_wrapper,
                _n_samples
            );
            ConsolidateCentroids();
            ComputeCost();
            ComputeShift();
            if (alpha == dp) {
                PostprocessCentroids();
            }
            if (n_queries) {
                _recall_time.Tic();
                float cur_recall = ComputeRecall(rotated_queries.data(), n_queries, objective_k, _centroids_to_explore);
                _recall_time.Toc();
                _recall = cur_recall;
            }
            if (verbose)
                std::cout << "Iteration " << iter_idx + 1 << "/" << _iters
                          << " | Objective: " << cost << " | Shift: " << shift
                          << " | Split: " << _n_split 
                          << " | Recall: " << _recall << std::endl << std::endl;
        }
        //! I only need assignments if sampling_faction < 1
        if (_sampling_fraction < 1.0f) {
            // TODO(@lkuffo, critical): Create proper assignments
            // Assign(data, centroids_pdx_wrapper, n);
        }

        _trained = true;
        if (verbose) {
            PrintTimes();
        }
        return _centroids;
    }

    void PrintTimes() {
        const float total_time =
            _total_search_time.accum_time + _allocator_time.accum_time + _rotator_time.accum_time +
            _norms_calc_time.accum_time + _sampling_time.accum_time + _reordering_time.accum_time +
            _total_centroids_update_time.accum_time + _grouping_time.accum_time +
            _centroids_splitting.accum_time + _pdxify_time.accum_time + _shift_time.accum_time;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::endl;
        std::cout << "TOTAL SEARCH TIME " << _total_search_time.accum_time / 1000000000.0 << " ("
                  << _total_search_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << " - BLAS " << _blas_total_time.accum_time / 1000000000.0 << " ("
                  << _blas_total_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << " - PDX  " << _pdx_search_time.accum_time / 1000000000.0 << " ("
                  << _pdx_search_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << " - NORMS  " << _blas_norms_time.accum_time / 1000000000.0 << " ("
                  << _blas_norms_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL ALLOCATOR TIME " << _allocator_time.accum_time / 1000000000.0 << " ("
                  << _allocator_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL ROTATOR TIME " << _rotator_time.accum_time / 1000000000.0 << " ("
                  << _rotator_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL NORMS CALCULATION TIME " << _norms_calc_time.accum_time / 1000000000.0
                  << " (" << _norms_calc_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL SAMPLING TIME " << _sampling_time.accum_time / 1000000000.0 << " ("
                  << _sampling_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL UPDATE CENTROIDS TIME "
                  << _total_centroids_update_time.accum_time / 1000000000.0 << " ("
                  << _total_centroids_update_time.accum_time / total_time * 100 << "%) "
                  << std::endl;
        std::cout << "TOTAL SPLITTING TIME " << _centroids_splitting.accum_time / 1000000000.0
                  << " (" << _centroids_splitting.accum_time / total_time * 100 << "%) "
                  << std::endl;
        std::cout << "TOTAL PDXIFYING TIME " << _pdxify_time.accum_time / 1000000000.0 << " ("
                  << _pdxify_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL SHIFT TIME " << _shift_time.accum_time / 1000000000.0 << " ("
                  << _shift_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL REORDERING TIME " << _reordering_time.accum_time / 1000000000.0 << " ("
                  << _reordering_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL GROUPING TIME " << _grouping_time.accum_time / 1000000000.0 << " ("
                  << _grouping_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL GT ASSIGNMENTS TIME " << _gt_assignments_time.accum_time / 1000000000.0 << " ("
                  << _gt_assignments_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL RECALL TIME " << _recall_time.accum_time / 1000000000.0 << " ("
                  << _recall_time.accum_time / total_time * 100 << "%) " << std::endl;
        std::cout << "TOTAL (s) "
                  << (_total_search_time.accum_time + _allocator_time.accum_time +
                      _rotator_time.accum_time + _norms_calc_time.accum_time +
                      _sampling_time.accum_time + _total_centroids_update_time.accum_time +
                      _centroids_splitting.accum_time + _pdxify_time.accum_time +
                      _shift_time.accum_time + _grouping_time.accum_time +
                      _reordering_time.accum_time + _gt_assignments_time.accum_time) /
                         1000000000.0
                  << std::endl;
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
        // TODO(@lkuffo, low): What if I start with a 1 to 1 distance with the same centroid that
        //      was assigned last time and I set it as an initial lower bound?
        //      I can force the Start() to that chunk?
        auto data_p = data;
        // TODO(@lkuffo, med): Skip the vectors that were used as samples
        for (size_t i = 0; i < n; ++i) {
            // PDXearch per vector
            std::vector<knn_candidate_t> assignment = pdx_centroids.searcher->Search(data_p, 1);
            auto assignment_idx = assignment[0].index;
            _assignments[i] = assignment_idx;
            data_p += _d;
        }
        // TODO(@lkuffo, medium): Do I need to return the true centroids?
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
        if (verbose) {
            std::cout << "Batch Calculation [START]..." << std::endl;
        }
        _blas_search_time.Reset();
        _blas_total_time.Tic();
        _blas_search_time.Tic();
        _total_search_time.Tic();
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
        _blas_search_time.Toc();
        _blas_total_time.Toc();
        _total_search_time.Toc();
        if (verbose)
            std::cout << "Total time for BLAS search (s): "
                      << _blas_search_time.accum_time / 1000000000.0 << std::endl;
        _all_search_time += _blas_search_time.accum_time / 1000000000.0;
        if (verbose) {
            std::cout << "Batch Calculation [DONE]..." << std::endl;
        }
        _sampling_time.Tic();
        std::fill(_tmp_centroids.begin(), _tmp_centroids.end(), 0.0);
        std::fill(_cluster_sizes.begin(), _cluster_sizes.end(), 0);
        _sampling_time.Toc();
        cost = 0.0;
        _centroids_update_time.Reset();
        _centroids_update_time.Tic();
        _total_centroids_update_time.Tic();
        UpdateCentroids(data, n);
        _centroids_update_time.Toc();
        _total_centroids_update_time.Toc();
        if (verbose)
            std::cout << "Total time for UpdateCentroid (s): "
                      << _centroids_update_time.accum_time / 1000000000.0 << std::endl;
    }

    template <bool RECORD_PRUNING_GROUP = false>
    void AssignAndUpdateCentroidsPartialBatched(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT partial_rotated_centroids,
        const vector_value_t* SKM_RESTRICT partial_data_norms,
        const vector_value_t* SKM_RESTRICT partial_centroid_norms,
        distance_t* SKM_RESTRICT all_distances,
        const layout_t& pdx_centroids,
        const size_t n
    ) {
        _sampling_time.Tic();
        std::copy(_tmp_centroids.begin(), _tmp_centroids.end(), _prev_centroids.begin());
        _sampling_time.Toc();

        cost = 0.0;
        _search_time.Reset();
        _search_time.Tic();
        _total_search_time.Tic();
        batch_computer::template Batched_XRowMajor_YRowMajor_MultiplePartialD<RECORD_PRUNING_GROUP>(
            data,
            partial_rotated_centroids,
            _prev_centroids.data(),
            n,
            _n_clusters,
            _d,
            partial_data_norms,
            partial_centroid_norms,
            _assignments.data(),
            _distances.data(),
            _pruning_groups.data(),
            _pruning_groups_partial_d.data(),
            _pruning_groups_ends.data(),
            all_distances,
            pdx_centroids,
            _blas_total_time,
            _pdx_search_time,
            _blas_norms_time,
            _initial_partial_d
        );
        _search_time.Toc();
        _total_search_time.Toc();
        _sampling_time.Tic();
        std::fill(_tmp_centroids.begin(), _tmp_centroids.end(), 0.0);
        std::fill(_cluster_sizes.begin(), _cluster_sizes.end(), 0);
        _sampling_time.Toc();
        if (verbose)
            std::cout << "Total time for BLAS+PDX search (s): "
                      << _search_time.accum_time / 1000000000.0 << std::endl;
        _all_search_time += _search_time.accum_time / 1000000000.0;

        _centroids_update_time.Reset();
        _centroids_update_time.Tic();
        _total_centroids_update_time.Tic();
        UpdateCentroids(data, n);
        _centroids_update_time.Toc();
        _total_centroids_update_time.Toc();
        if (verbose)
            std::cout << "Total time for UpdateCentroid (s): "
                      << _centroids_update_time.accum_time / 1000000000.0 << std::endl;
    }

    void UpdateCentroids(const vector_value_t* SKM_RESTRICT data, const size_t n) {
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
            _tmp_centroids[cluster_idx * _d + i] += vector[i];
        }
    }

    void SplitClusters() {
        _n_split = 0;
        std::default_random_engine rng;
        auto _tmp_centroids_p = _tmp_centroids.data();
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
                    (void*) (_tmp_centroids_p + ci * _d),
                    (void*) (_tmp_centroids_p + cj * _d),
                    sizeof(centroid_value_t) * _d
                );

                // Small symmetric perturbation
                for (size_t j = 0; j < _d; j++) {
                    if (j % 2 == 0) {
                        _tmp_centroids_p[ci * _d + j] *= 1 + CENTROID_PERTURBATION_EPS;
                        _tmp_centroids_p[cj * _d + j] *= 1 - CENTROID_PERTURBATION_EPS;
                    } else {
                        _tmp_centroids_p[ci * _d + j] *= 1 - CENTROID_PERTURBATION_EPS;
                        _tmp_centroids_p[cj * _d + j] *= 1 + CENTROID_PERTURBATION_EPS;
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
        _centroids_splitting.Tic();
#pragma omp parallel for if (g_n_threads > 1) num_threads(g_n_threads)
        for (size_t i = 0; i < _n_clusters; ++i) {
            auto _tmp_centroids_p = _tmp_centroids.data() + i * _d;
            if (_cluster_sizes[i] == 0) {
                continue;
            }
            float mult_factor = 1.0 / _cluster_sizes[i];
#pragma clang loop vectorize(enable)
            for (size_t j = 0; j < _d; ++j) {
                _tmp_centroids_p[j] *= mult_factor;
            }
        }
        SplitClusters();
        _centroids_splitting.Toc();
        _pdxify_time.Tic();
        //! This updates the object within the pdx_layout wrapper
        PDXLayout<q, alpha, Pruner>::template PDXify<false>(
            _tmp_centroids.data(), _centroids.data(), _n_clusters, _d
        );
        CentroidsToAuxiliaryHorizontal();
        _pdxify_time.Toc();
    }

    void ComputeCost() {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < _n_samples; ++i) {
            cost += _distances[i];
        }
    }

    void ComputeShift() {
        _shift_time.Tic();
        Eigen::Map<const MatrixR> new_mat(_tmp_centroids.data(), _n_clusters, _d);
        Eigen::Map<const MatrixR> prev_mat(_prev_centroids.data(), _n_clusters, _d);
        MatrixR diff = new_mat - prev_mat;
        shift = 0.0f;
#pragma omp parallel for reduction(+:shift) num_threads(g_n_threads)
        for (size_t i = 0; i < _n_clusters; ++i) {
            shift += diff.row(i).squaredNorm();
        }
        shift /= (_n_clusters * _d);
        _shift_time.Toc();
    }

    void GetGTAssignmentsAndDistances(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        const vector_value_t* SKM_RESTRICT queries,
        const size_t n_queries,
        const size_t objective_k
    ) {
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
        std::vector<distance_t> tmp_distances_buffer(X_BATCH_SIZE * Y_BATCH_SIZE);
        // TODO(@lkuffo, crit): Everytime we call this function we compute norms again and again.
        std::vector<distance_t> query_norms(n_queries);
        GetL2NormsRowMajor(queries, n_queries, query_norms.data());
        std::vector<uint32_t> promising_centroids(n_queries * centroids_to_explore);
        std::vector<distance_t> distances(n_queries * centroids_to_explore);
        batch_computer::Batched_XRowMajor_YRowMajor_TopK(
            queries,
            _tmp_centroids.data(),
            n_queries,
            _n_clusters,
            _d,
            query_norms.data(),
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
        _sampling_time.Tic();
        auto tmp_centroids_p = _tmp_centroids.data();
        // Equidistant sampling similar to DuckDB's
        // const auto jumps = static_cast<size_t>(std::floor(1.0 * n / _n_clusters));
        // for (size_t i = 0; i < n; i += jumps) {
        //     // TODO(@lkuffo, low): What if centroid scalar_t are not the same size of vector ones
        //     memcpy(
        //         (void*) tmp_centroids_p, (void*) (data + (i * _d)), sizeof(centroid_value_t) * _d
        //     );
        //     tmp_centroids_p += _d;
        // }
        // First `n` samples similar to FAISS'
        for (size_t i = 0; i < _n_clusters; i += 1) {
            // TODO(@lkuffo, low): What if centroid scalar_t are not the same size of vector ones
            memcpy(
                (void*) tmp_centroids_p, (void*) (data + (i * _d)), sizeof(centroid_value_t) * _d
            );
            tmp_centroids_p += _d;
        }
        _sampling_time.Toc();
        // We populate the _centroids buffer with the centroids in the PDX layout
        _allocator_time.Tic();
        std::vector<centroid_value_t> rotated_centroids(_n_clusters * _d);
        _allocator_time.Toc();
        _rotator_time.Tic();
        _pruner->Rotate(_tmp_centroids.data(), rotated_centroids.data(), _n_clusters);
        _rotator_time.Toc();
        _pdxify_time.Tic();
        PDXLayout<q, alpha, Pruner>::template PDXify<false>(
            rotated_centroids.data(), _centroids.data(), _n_clusters, _d
        );
        _pdxify_time.Toc();
        _sampling_time.Tic();
        //! We wrap _centroids and _aux_hor_centroids in the PDXLayout wrapper
        //! Any updates to these objects is reflected in the PDXLayout
        auto pdx_centroids = PDXLayout<q, alpha, Pruner>(
            _centroids.data(), *_pruner, _n_clusters, _d, _aux_hor_centroids.data()
        );
        _sampling_time.Toc();
        return pdx_centroids;
    }

    /*
     * This function creates groups with a similar pruning thresholds.
     * For now, we aim for 5 or 6 groups with around the same amount of items
     * The pruning thresholds are found in `groups`
     */
    void CreatePruningGroups(float* SKM_RESTRICT data, const size_t n) {
        _grouping_time.Tic();
        _pruning_groups_ends.clear();
        _pruning_groups_partial_d.clear();
        std::cout << "Creating pruning groups" << std::endl;
        std::cout << "N: " << n << std::endl;
        const uint32_t approx_group_size =
            CeilXToMultipleOfM((n + (_n_pruning_groups - 1)) / _n_pruning_groups, X_BATCH_SIZE);
        std::cout << "Approx group size: " << approx_group_size << std::endl;
        const uint32_t min_group_size =
            CeilXToMultipleOfM(approx_group_size * (1 - PRUNING_GROUP_DEVIATION_FACTOR), X_BATCH_SIZE);
        std::cout << "Minimum group size: " << min_group_size << std::endl;

        // If we have less than a certain amount of tuples, we just go with one group
        if (n < X_BATCH_SIZE * _n_pruning_groups) {
            _pruning_groups_ends.push_back(n);
            _pruning_groups_partial_d.push_back(_vertical_d);
            return;
        }

        std::vector<uint32_t> new_ordering(n);
        std::iota(new_ordering.begin(), new_ordering.end(), 0);

        // Sort both _pruning_groups values and the indices for later _memcpy
        std::stable_sort(new_ordering.begin(), new_ordering.end(), [&](uint32_t a, uint32_t b) {
            return _pruning_groups[a] < _pruning_groups[b];
        });
        std::vector<uint32_t> _pruning_groups_sorted(n);
        for (size_t i = 0; i < n; ++i) {
            _pruning_groups_sorted[i] = _pruning_groups[new_ordering[i]];
        }
        std::copy(
            _pruning_groups_sorted.begin(), _pruning_groups_sorted.end(), _pruning_groups.begin()
        );

        uint32_t running_group_size = 0;
        uint32_t running_group_d = _pruning_groups[0];
        uint32_t cur_group = running_group_d;
        for (size_t i = 0; i < n; ++i) {
            cur_group = _pruning_groups[i];
            if (running_group_size >= approx_group_size) { // We have a full group
                _pruning_groups_ends.push_back(i);
                _pruning_groups_partial_d.push_back(running_group_d);
                running_group_size = 0;
                // !The next group could still have the same `d`, that is fine
                running_group_d = cur_group;
            }
            if (cur_group == running_group_d) { // Same group, still not full
                running_group_size += 1;
            } else { // Group has changed
                // Check if current group is approximately full
                if (running_group_size > min_group_size) {
                    _pruning_groups_ends.push_back(i);
                    _pruning_groups_partial_d.push_back(running_group_d);
                    running_group_d = cur_group;
                    running_group_size = 1;
                } else {
                    // We fuse the current group to the next `d`
                    running_group_d = cur_group;
                    running_group_size += 1;
                }
            }
        }
        std::cout << "Pruning Groups before Last correction" << std::endl;
        for (size_t i = 0; i < _pruning_groups_ends.size(); ++i) {
            std::cout << _pruning_groups_partial_d[i] << ": " << _pruning_groups_ends[i]
                      << std::endl;
        }
        // Remaining group
        if (running_group_size > 0) {
            running_group_d = _pruning_groups_partial_d.back();
            std::cout << "Cur group" << ": " << cur_group << std::endl;
            std::cout << "Last running group" << ": " << running_group_d << std::endl;
            // We don't want to fuse a `d` group with a `d*2` group and redefine them under `d*2`
            // If the remainder is small or the last group is still running we fuse with previous
            if (running_group_size <= X_BATCH_SIZE || cur_group == running_group_d) {
                _pruning_groups_ends.back() = static_cast<uint32_t>(n);
            } else { // Otherwise, we create a new group. This would be the tail of the plot
                _pruning_groups_ends.push_back(n);
                std::cout << "N: " << n << std::endl;
                std::cout << "N? " << _pruning_groups_ends.back() << std::endl;
                _pruning_groups_partial_d.push_back(cur_group);
                running_group_size = 0;
            }
        }

        // Make sure every group except the last one is of a size multiple of X_BATCH_SIZE
        if (_pruning_groups_ends.size() <= 1) {
            assert(_pruning_groups_ends.size() == _pruning_groups_partial_d.size());
            assert(_pruning_groups_ends.back() == n);
            return;
        }
        std::cout << "Pruning Groups Before correction" << std::endl;
        for (size_t i = 0; i < _pruning_groups_ends.size(); ++i) {
            std::cout << _pruning_groups_partial_d[i] << ": " << _pruning_groups_ends[i]
                      << std::endl;
        }
        for (size_t i = 0; i + 1 < _pruning_groups_ends.size(); ++i) {
            size_t cur_end = _pruning_groups_ends[i];
            size_t prev_end = (i == 0) ? 0 : _pruning_groups_ends[i - 1];
            size_t cur_group_size = cur_end - prev_end;
            if (cur_group_size % X_BATCH_SIZE != 0) {
                // We want to send lower pruning groups to higher pruning groups, not the other way
                auto new_group_size = FloorXToMultipleOfM(cur_group_size, X_BATCH_SIZE);
                assert(new_group_size < cur_group_size);
                auto n_send_to_next = cur_group_size - new_group_size;
                _pruning_groups_ends[i] = (prev_end + new_group_size);
                if (i + 1 == _pruning_groups_ends.size()) { // We don't want to change the last val
                    _pruning_groups_ends[i + 1] = (_pruning_groups_ends[i + 1] + n_send_to_next);
                }
            }
        }
        std::cout << "Pruning Groups" << std::endl;
        for (size_t i = 0; i < _pruning_groups_ends.size(); ++i) {
            std::cout << _pruning_groups_partial_d[i] << ": " << _pruning_groups_ends[i]
                      << std::endl;
        }

        assert(_pruning_groups_ends.size() == _pruning_groups_partial_d.size());
        assert(_pruning_groups_ends.back() == n);

        _grouping_time.Toc();
        std::cout << "Permuting rows" << std::endl;
        _reordering_time.Tic();
        PermuteMatrixRows(data, n, new_ordering.data());
        _reordering_time.Toc();
        std::cout << "End Permuting rows" << std::endl;
    }

    void PermuteMatrixRows(float* SKM_RESTRICT data, const size_t n, const uint32_t* new_ordering) {
        std::vector<bool> visited(n, false);
        std::vector<float> tmp_row(_d); // temporary row buffer
        auto tmp_assignment = _assignments[0];
        auto tmp_distances = _distances[0];
        for (size_t i = 0; i < n; ++i) {
            if (visited[i] || new_ordering[i] == i)
                continue; // already in place or processed
            size_t j = i;
            // copy the first row in the cycle
            memcpy(tmp_row.data(), data + j * _d, sizeof(vector_value_t) * _d);
            tmp_assignment = _assignments[j];
            tmp_distances = _distances[j];
            while (!visited[j]) {
                visited[j] = true;
                size_t next = new_ordering[j];
                if (next == i) {
                    // end of cycle: put tmp_row into the final position
                    memcpy(data + j * _d, tmp_row.data(), sizeof(vector_value_t) * _d);
                    _assignments[j] = tmp_assignment;
                    _distances[j] = tmp_distances;
                } else {
                    // move the row from `next` into position `j`
                    memcpy(data + j * _d, data + next * _d, sizeof(vector_value_t) * _d);
                    _assignments[j] = _assignments[next];
                    _distances[j] = _distances[next];
                }
                j = next;
            }
        }
    }

    void GetGroupsL2NormsRowMajor(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        vector_value_t* SKM_RESTRICT out_norm
    ) {
        _norms_calc_time.Tic();
        Eigen::Map<const MatrixR> e_data(data, n, _d);
        Eigen::Map<VectorR> e_norms(out_norm, n);
        const auto n_groups = _pruning_groups_partial_d.size();
        if (n_groups == 1) {
            const uint32_t pd = _pruning_groups_partial_d[0];
            e_norms.noalias() = e_data.leftCols(pd).rowwise().squaredNorm();
        } else {
            size_t start = 0;
            for (size_t i = 0; i < n_groups; ++i) {
                const size_t end = _pruning_groups_ends[i];
                const size_t count = end - start;
                const uint32_t pd = _pruning_groups_partial_d[i];
                assert(end <= n);
                auto block = e_data.block(start, 0, count, pd);
                e_norms.segment(start, count).noalias() = block.rowwise().squaredNorm();
                start = end; // move to next group
            }
        }
        _norms_calc_time.Toc();
    }

    void GetL2NormsRowMajorPerGroup(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        vector_value_t* SKM_RESTRICT out_norm
    ) {
        _norms_calc_time.Tic();
        const auto n_groups = _pruning_groups_partial_d.size();
        auto out_norm_p = out_norm;
        Eigen::Map<const MatrixR> e_data(data, n, _d);
        for (size_t i = 0; i < n_groups; ++i) {
            const auto partial_d = _pruning_groups_partial_d[i];
            Eigen::Map<VectorR> e_norms(out_norm_p, n);
            e_norms.noalias() = e_data.leftCols(partial_d).rowwise().squaredNorm();
            out_norm_p += n;
        }
        _norms_calc_time.Toc();
    }

    void GetL2NormsRowMajor(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        vector_value_t* SKM_RESTRICT out_norm
    ) {
        _norms_calc_time.Tic();
        Eigen::Map<const MatrixR> e_data(data, n, _d);
        Eigen::Map<VectorR> e_norms(out_norm, n);
        e_norms.noalias() = e_data.rowwise().squaredNorm();
        _norms_calc_time.Toc();
    }

    void CentroidsToAuxiliaryHorizontal() {
        Eigen::Map<MatrixR> hor_centroids(_tmp_centroids.data(), _n_clusters, _d);
        Eigen::Map<MatrixR> out_aux_centroids(_aux_hor_centroids.data(), _n_clusters, _vertical_d);
        out_aux_centroids.noalias() = hor_centroids.leftCols(_vertical_d);
    }

    void GetL2NormsRowMajor(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        vector_value_t* SKM_RESTRICT out_norm,
        const size_t partial_d
    ) {
        _norms_calc_time.Tic();
        Eigen::Map<const MatrixR> e_data(data, n, _d);
        Eigen::Map<VectorR> e_norms(out_norm, n);
        e_norms.noalias() = e_data.leftCols(partial_d).rowwise().squaredNorm();
        _norms_calc_time.Toc();
    }

    size_t GetNVectorsToSample(const size_t n) const {
        if (_sampling_fraction == 1.0) {
            return n;
        }
        return std::floor(n * _sampling_fraction);
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

    // Equidistant sampling similar to DuckDB's
    template <bool ROTATE = true>
    vector_value_t* SampleVectors(
        const vector_value_t* SKM_RESTRICT data,
        std::vector<vector_value_t>& data_samples_buffer,
        const size_t n,
        const size_t n_samples
    ) { // TODO(@lkuffo, med): can be const function
        const vector_value_t* tmp_data_buffer_p = nullptr;
        std::vector<vector_value_t> samples_tmp;
        // TODO(@lkuffo, medium): If DP, normalize here while taking the samples
        _sampling_time.Tic();
        if (n_samples < n) {
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
            tmp_data_buffer_p = samples_tmp.data();
        } else {
            tmp_data_buffer_p = data; // Zero-copy
        }
        _sampling_time.Toc();

        std::cout << "n_samples: " << n_samples << std::endl;

        // TODO(@lkuffo, crit): This buffer is a headache
            // Sometimes I would not need to rotate
        _allocator_time.Tic();
        data_samples_buffer.resize(n_samples * _d);
        _allocator_time.Toc();
        _rotator_time.Tic();
        std::cout << "Rotating 1..." << std::endl;
        if (ROTATE) {
            _pruner->Rotate(tmp_data_buffer_p, data_samples_buffer.data(), n_samples);
        } else {
            memcpy(
                (void*) data_samples_buffer.data(),
                (void*) tmp_data_buffer_p,
                sizeof(vector_value_t) * n_samples * _d
            );
        }
        _rotator_time.Toc();
        return data_samples_buffer.data();
    }

    std::unique_ptr<Pruner> _pruner;

    std::vector<centroid_value_t> _centroids;      // Always keeps the PDX centroids
    std::vector<centroid_value_t> _tmp_centroids;  // Always keeps the horizontal centroids
    std::vector<centroid_value_t> _prev_centroids; // Always keeps the previous iteration centroids
    std::vector<centroid_value_t> _aux_hor_centroids;
    std::vector<uint32_t> _assignments;
    std::vector<distance_t> _distances;
    std::vector<uint32_t> _cluster_sizes;

    std::vector<uint32_t> _gt_assignments;
    std::vector<distance_t> _gt_distances;

    std::vector<uint32_t> _pruning_groups;
    std::vector<uint32_t> _pruning_groups_ends;
    std::vector<uint32_t> _pruning_groups_partial_d;

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
    uint32_t _n_pruning_groups = 1;
    uint32_t _vertical_d;
    float tol;
    float _recall = 0.0f;
    size_t _centroids_to_explore = 64;

    TicToc _grouping_time;
    TicToc _reordering_time;
    TicToc _search_time;
    TicToc _blas_search_time;
    TicToc _blas_total_time;
    TicToc _centroids_update_time;
    TicToc _total_centroids_update_time;
    TicToc _centroids_splitting;
    TicToc _allocator_time;
    TicToc _rotator_time;
    TicToc _sampling_time;
    TicToc _norms_calc_time;
    TicToc _total_search_time;
    TicToc _pdxify_time;
    TicToc _pdx_search_time;
    TicToc _shift_time;
    TicToc _blas_norms_time;
    TicToc _gt_assignments_time;
    TicToc _recall_time;
    float _all_search_time = 0.0;
};
} // namespace skmeans
