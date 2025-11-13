#pragma once
#ifndef SUPERKMEANS_H
#define SUPERKMEANS_H

#include <Eigen/Eigen/Dense>
#include <iomanip>
#include <omp.h>
#include <random>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.hpp"
#include "superkmeans/distance_computers/batch_computers.hpp"
#include "superkmeans/pdx/pdxearch.h"
#include "superkmeans/pdx/utils.h"

#define EPS (1 / 1024.)

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
    // TODO(@lkuffo, high): N-threads should be controlled with a global function. By default
    //   it is set to all threads
    SuperKMeans(
        size_t n_clusters,
        size_t dimensionality,
        uint32_t iters = 25,
        float sampling_fraction = 0.50,
        bool verbose = false,
        uint32_t n_threads = 1
    )
        : _iters(iters), _n_clusters(n_clusters), _sampling_fraction(sampling_fraction),
          _d(dimensionality), _trained(false), verbose(verbose), N_THREADS(n_threads) {
        SKMEANS_ENSURE_POSITIVE(n_clusters);
        SKMEANS_ENSURE_POSITIVE(iters);
        SKMEANS_ENSURE_POSITIVE(sampling_fraction);
        SKMEANS_ENSURE_POSITIVE(dimensionality);
        // TODO(@lkuffo, low): Support non-multiples of 64
        assert(n_clusters % 64 == 0);
        if (sampling_fraction > 1.0) {
            throw std::invalid_argument("sampling_fraction must be smaller than 1");
        }
        _pruner = std::make_unique<Pruner>(dimensionality, 1.5);
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
        uint32_t num_threads = -1
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
        std::vector<vector_value_t> data_norms(_n_samples);
        std::vector<vector_value_t> centroid_norms(_n_clusters);
        std::vector<distance_t> all_distances(X_BATCH_SIZE * Y_BATCH_SIZE);
        std::vector<uint32_t> out_knn(n);
        std::vector<distance_t> out_distances(n);
        _allocator_time.Toc();

        // TODO(@lkuffo, med): If metric is dp, normalize the vectors so we can use l2

        if (verbose) {
            std::cout << "Generating centroids..." << std::endl;
        }
        //! _centroids is always wrapped with the PDXLayout object
        auto centroids_pdx_wrapper = GenerateCentroids(data_p, n);
        if (verbose) {
            std::cout << "Sampling data..." << std::endl;
        }

        std::vector<vector_value_t> data_samples_buffer;
        auto data_to_cluster = SampleVectors(data_p, data_samples_buffer, n);

        // TODO(@lkuffo, low): I don't like this rotated_initial_centroids variable
        _rotator_time.Tic();
        std::vector<centroid_value_t> rotated_initial_centroids(_n_clusters * _d);
        _pruner->Rotate(_tmp_centroids.data(), rotated_initial_centroids.data(), _n_clusters);
        _rotator_time.Toc();
        GetL2NormsRowMajor(data_to_cluster, _n_samples, data_norms.data());
        GetL2NormsRowMajor(rotated_initial_centroids.data(), _n_clusters, centroid_norms.data());

        InitAssignAndUpdateCentroids(
            data_to_cluster,
            rotated_initial_centroids.data(),
            data_norms.data(),
            centroid_norms.data(),
            all_distances.data(),
            out_knn.data(),
            out_distances.data(),
            _n_samples
        );
        size_t iter_idx = 1;
        ConsolidateCentroids();
        ComputeCost();

        if (verbose)
            std::cout << "Iteration 1" << "/" << _iters << " | Objective: " << cost
                      << " | Split: " << _n_split << std::endl
                      << std::endl;
        GetL2NormsRowMajor(
            data_to_cluster, _n_samples, data_norms.data(), _partial_d
        ); // BATCHED ITER
        for (; iter_idx < _iters; ++iter_idx) {
            // BATCHED ITER
            GetL2NormsRowMajor(
                _tmp_centroids.data(), _n_clusters, centroid_norms.data(), _partial_d
            );
            AssignAndUpdateCentroidsPartialBatched(
                data_to_cluster,
                _tmp_centroids.data(),
                data_norms.data(),
                centroid_norms.data(),
                all_distances.data(),
                centroids_pdx_wrapper,
                _n_samples
            );

            // NORMAL
            // if (verbose) {
            //     std::cout << "Assigning..." << std::endl;
            // }
            // AssignAndUpdateCentroids(data_to_cluster, centroids_pdx_wrapper, _n_samples);

            ConsolidateCentroids();
            ComputeCost();
            if (alpha == dp) {
                PostprocessCentroids();
            }
            if (verbose)
                std::cout << "Iteration " << iter_idx + 1 << "/" << _iters
                          << " | Objective: " << cost << " | Split: " << _n_split << std::endl
                          << std::endl;
        }
        //! I don't need proper assignments until the last iteration
        if (_sampling_fraction == 1) {
        }
        // TODO(@lkuffo, critical): Create proper assignments
        // Assign(data, centroids_pdx_wrapper, n);

        _trained = true;
        if (verbose) {
            const float total_time = _total_search_time.accum_time + _allocator_time.accum_time +
                                     _rotator_time.accum_time + _norms_calc_time.accum_time +
                                     _sampling_time.accum_time +
                                     _total_centroids_update_time.accum_time +
                                     _centroids_splitting.accum_time + _pdxify_time.accum_time;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << std::endl;
            std::cout << "TOTAL SEARCH TIME " << _total_search_time.accum_time / 1000000000.0
                      << " (" << _total_search_time.accum_time / total_time * 100 << "%) "
                      << std::endl;
            std::cout << " - BLAS " << _blas_total_time.accum_time / 1000000000.0 << " ("
                      << _blas_total_time.accum_time / total_time * 100 << "%) " << std::endl;
            std::cout << " - PDX  " << _pdx_search_time.accum_time / 1000000000.0 << " ("
                      << _pdx_search_time.accum_time / total_time * 100 << "%) " << std::endl;
            std::cout << "TOTAL ALLOCATOR TIME " << _allocator_time.accum_time / 1000000000.0
                      << " (" << _allocator_time.accum_time / total_time * 100 << "%) "
                      << std::endl;
            std::cout << "TOTAL ROTATOR TIME " << _rotator_time.accum_time / 1000000000.0 << " ("
                      << _rotator_time.accum_time / total_time * 100 << "%) " << std::endl;
            std::cout << "TOTAL NORMS CALCULATION TIME "
                      << _norms_calc_time.accum_time / 1000000000.0 << " ("
                      << _norms_calc_time.accum_time / total_time * 100 << "%) " << std::endl;
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
            std::cout << "TOTAL (s) "
                      << (_total_search_time.accum_time + _allocator_time.accum_time +
                          _rotator_time.accum_time + _norms_calc_time.accum_time +
                          _sampling_time.accum_time + _total_centroids_update_time.accum_time +
                          _centroids_splitting.accum_time + _pdxify_time.accum_time) /
                             1000000000.0
                      << std::endl;
        }
        return _centroids;
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
        uint32_t* SKM_RESTRICT out_knn,
        distance_t* SKM_RESTRICT out_distances,
        const size_t n
    ) {
        if (verbose) {
            std::cout << "Batch Calculation [START]..." << std::endl;
        }
        _blas_search_time.Reset();
        _blas_total_time.Tic();
        _blas_search_time.Tic();
        _total_search_time.Tic();
        // batch_computer::Batch_XRowMajor_YRowMajor(
        //     data,
        //     rotated_initial_centroids,
        //     n,
        //     _n_clusters,
        //     _d,
        //     data_norms,
        //     centroid_norms,
        //     out_knn,
        //     out_distances,
        //     all_distances
        // );
        batch_computer::Batched_XRowMajor_YRowMajor(
            data,
            rotated_initial_centroids,
            n,
            _n_clusters,
            _d,
            data_norms,
            centroid_norms,
            out_knn,
            out_distances,
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
#pragma omp parallel for if (N_THREADS > 1) num_threads(N_THREADS)
        for (size_t i = 0; i < n; ++i) {
            auto assignment_idx = out_knn[i];
            auto assignment_distance = out_distances[i];
            _assignments[i] = assignment_idx;
            _distances[i] = assignment_distance;
        }
        UpdateCentroids(data, n);
        _centroids_update_time.Toc();
        _total_centroids_update_time.Toc();
        if (verbose)
            std::cout << "Total time for UpdateCentroid (s): "
                      << _centroids_update_time.accum_time / 1000000000.0 << std::endl;
    }

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
        _pdx_search_time.Tic();
        _total_search_time.Tic();
        batch_computer::Batched_XRowMajor_YRowMajor_PartialD(
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
            all_distances,
            pdx_centroids,
            _partial_d
        );
        _search_time.Toc();
        _pdx_search_time.Toc();
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

    // TODO(lkuffo, high): If less than 128 dimensions, then do not even attempt to prune.
    //   Just send everything to batched queries
    void AssignAndUpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const layout_t& pdx_centroids,
        const size_t n
    ) {
        _sampling_time.Tic();
        std::copy(_tmp_centroids.begin(), _tmp_centroids.end(), _prev_centroids.begin());
        std::fill(_tmp_centroids.begin(), _tmp_centroids.end(), 0.0);
        std::fill(_cluster_sizes.begin(), _cluster_sizes.end(), 0);
        _sampling_time.Toc();
        // auto data_p = data;
        cost = 0.0;
        _search_time.Reset();
        _search_time.Tic();
        _pdx_search_time.Tic();
        _total_search_time.Tic();
        // TODO(@lkuffo, crit): Fork Union
#pragma omp parallel for if (N_THREADS > 1) num_threads(N_THREADS)
        for (size_t i = 0; i < n; ++i) {
            const auto data_p = data + i * _d;
            // We calculate the distance from the previous assignment
            const auto prev_assignment = _assignments[i];
            const distance_t dist_to_prev_centroid = DistanceComputer<alpha, q>::Horizontal(
                _prev_centroids.data() + (prev_assignment * _d), data_p, _d
            );
            // PDXearch per vector
            std::vector<knn_candidate_t> assignment =
                pdx_centroids.searcher->Top1SearchWithThreshold(
                    data_p, dist_to_prev_centroid, prev_assignment, i
                );
            auto [assignment_idx, assignment_distance] = assignment[0];
            _assignments[i] = assignment_idx;
            _distances[i] = assignment_distance;
        }
        _total_search_time.Toc();
        _search_time.Toc();
        _pdx_search_time.Toc();
        if (verbose)
            std::cout << "Total time for PDX search (s): " << _search_time.accum_time / 1000000000.0
                      << std::endl;
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
#pragma omp parallel num_threads(N_THREADS)
        {
            uint32_t nt = N_THREADS;
            uint32_t rank = omp_get_thread_num();
            // This thread is taking care of centroids c0:c1
            size_t c0 = (_n_clusters * rank) / nt;
            size_t c1 = (_n_clusters * (rank + 1)) / nt;

            // OLD LOOP
            // for (size_t i = 0; i < n; ++i) {
            //     const auto data_p = data + i * _d;
            //     const auto assignment_idx = _assignments[i];
            //     UpdateCentroid(data_p, assignment_idx);
            // }
            ////////////

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
        // Potentially also store the information to quickly calculate the norms for DP
        // TODO(@lkuffo, low): This should be trivially auto-vectorized in any architecture
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
                        _tmp_centroids_p[ci * _d + j] *= 1 + EPS;
                        _tmp_centroids_p[cj * _d + j] *= 1 - EPS;
                    } else {
                        _tmp_centroids_p[ci * _d + j] *= 1 - EPS;
                        _tmp_centroids_p[cj * _d + j] *= 1 + EPS;
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
        for (size_t i = 0; i < _n_clusters; ++i) {
            _reciprocal_cluster_sizes[i] = 1.0 / _cluster_sizes[i];
        }
        auto _tmp_centroids_p = _tmp_centroids.data();
        for (size_t i = 0; i < _n_clusters; ++i) {
            if (_cluster_sizes[i] == 0) {
                _tmp_centroids_p += _d;
                continue;
            }
            auto mult_factor = _reciprocal_cluster_sizes[i];
            // TODO(@lkuffo, low): This should be trivially auto-vectorized in any architecture
#pragma clang loop vectorize(enable)
            for (size_t j = 0; j < _d; ++j) {
                _tmp_centroids_p[j] = _tmp_centroids_p[j] * mult_factor;
            }
            _tmp_centroids_p += _d;
        }
        SplitClusters();
        _centroids_splitting.Toc();
        // memcpy and PDXify in one go?
        // TODO(@lkuffo, high): Pruning flag false/true
        _pdxify_time.Tic();
        PDXLayout<q, alpha, Pruner>::template PDXify<false>(
            _tmp_centroids.data(), _centroids.data(), _n_clusters, _d
        );
        _pdxify_time.Toc();
    }

    void ComputeCost() {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < _n_samples; ++i) {
            cost += _distances[i];
        }
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
        const auto jumps = static_cast<size_t>(std::floor(1.0 * n / _n_clusters));
        auto tmp_centroids_p = _tmp_centroids.data();
        // Equidistant sampling similar to DuckDB's
        for (size_t i = 0; i < n; i += jumps) {
            // TODO(@lkuffo, low): What if centroid scalar_t are not the same size of vector ones
            memcpy(
                (void*) tmp_centroids_p, (void*) (data + (i * _d)), sizeof(centroid_value_t) * _d
            );
            tmp_centroids_p += _d;
        }
        _sampling_time.Toc();
        // We populate the _centroids buffer with the centroids in the PDX layout
        if constexpr (std::is_same_v<Pruner, ADSamplingPruner<q>>) {
            // TODO(@lkuffo, high): Implement a template bool for pruning or not, this would depend
            //    on the dimensionality of the data
            _rotator_time.Tic();
            std::vector<centroid_value_t> rotated_centroids(_n_clusters * _d);
            _pruner->Rotate(_tmp_centroids.data(), rotated_centroids.data(), _n_clusters);
            _rotator_time.Toc();
            _pdxify_time.Tic();
            PDXLayout<q, alpha, Pruner>::template PDXify<false>(
                rotated_centroids.data(), _centroids.data(), _n_clusters, _d
            );
            _pdxify_time.Toc();
        } else {
            PDXLayout<q, alpha, Pruner>::template PDXify<true>(
                tmp_centroids_p.data(), _centroids.data(), _n_clusters, _d
            );
        }
        _sampling_time.Tic();
        //! We wrap _centroids in the PDXLayout wrapper
        auto pdx_centroids =
            PDXLayout<q, alpha, Pruner>(_centroids.data(), *_pruner, _n_clusters, _d);
        _sampling_time.Toc();
        return pdx_centroids;
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
        if (_sampling_fraction == 1.0) { // Needed?
            return n;
        }
        return std::floor(n * _sampling_fraction);
    }

    // TODO(@lkuffo, high): Centroids are on PDX, I cant do this... but TMP centroids are not!
    void PostprocessCentroids() {
        auto centroids_p = _centroids.data();
        for (size_t i = 0; i < _n_clusters; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < _d; ++j) {
                sum += centroids_p[j] * centroids_p[j];
            }
            float norm = std::sqrt(sum);
            for (size_t j = 0; j < _d; ++i) {
                centroids_p[j] = centroids_p[j] / norm;
            }
        }
    }

    // Equidistant sampling similar to DuckDB's
    vector_value_t* SampleVectors(
        const vector_value_t* SKM_RESTRICT data,
        std::vector<vector_value_t>& data_samples_buffer,
        const size_t n
    ) { // TODO(@lkuffo, med): can be const function
        const vector_value_t* tmp_data_buffer_p = nullptr;
        std::vector<vector_value_t> samples_tmp;
        // TODO(@lkuffo, medium): If DP, normalize here while taking the samples
        _sampling_time.Tic();
        if (_n_samples < n) {
            samples_tmp.resize(_n_samples * _d);
            const auto jumps = static_cast<size_t>(std::floor((1.0 * n) / _n_samples));
            for (size_t i = 0; i < _n_samples; i += jumps) {
                memcpy(
                    (void*) (samples_tmp.data() + (_d * i)),
                    (void*) (data + i),
                    sizeof(vector_value_t) * _d
                );
            }
            tmp_data_buffer_p = samples_tmp.data();
        } else {
            tmp_data_buffer_p = data; // Zero-copy
        }
        _sampling_time.Toc();

        std::cout << "_n_samples: " << _n_samples << std::endl;
        if constexpr (std::is_same_v<Pruner, ADSamplingPruner<q>>) {
            // TODO(@lkuffo, high): Try to remove temporary buffer for rotating the vectors (sad)
            _rotator_time.Tic();
            data_samples_buffer.resize(_n_samples * _d);
            _pruner->Rotate(tmp_data_buffer_p, data_samples_buffer.data(), _n_samples);
            _rotator_time.Toc();
            return data_samples_buffer.data();
        } else {                      // Flat
            return tmp_data_buffer_p; // Zero-copy
        }
    }

    std::unique_ptr<Pruner> _pruner;

    std::vector<centroid_value_t> _centroids;
    std::vector<centroid_value_t> _tmp_centroids;
    std::vector<centroid_value_t> _prev_centroids;
    std::vector<centroid_value_t> _super_centroids;
    std::vector<uint32_t> _assignments;
    std::vector<distance_t> _distances;
    std::vector<uint32_t> _cluster_sizes;
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
    size_t _n_samples;
    size_t _n_split;
    uint32_t N_THREADS;

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
    float _all_search_time = 0.0;
    uint32_t _partial_d = 256;
};
} // namespace skmeans

#endif // SUPERKMEANS_H
