#pragma once

#include <Eigen/Eigen/Dense>
#include <iomanip>
#include <omp.h>
#include <random>

#include "distance_computers/gpu_batch_computers.h"
#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/distance_computers/gpu_batch_computers.h"
#include "superkmeans/pdx/pdxearch.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/profiler.h"
#include "superkmeans/gpu/gpu.cuh"

namespace skmeans {

/**
 * @brief Configuration parameters for SuperKMeans clustering.
 *
 */
struct SuperKMeansConfig {
    // Training parameters
    uint32_t iters = 25;            ///< Number of k-means iterations
    float sampling_fraction = 1.0f; ///< Fraction of data to sample (0.0 to 1.0)
    uint32_t n_threads = 0;         ///< Number of CPU threads (0 = auto-detect max)
    uint32_t seed = 42;             ///< Random seed for reproducibility
    bool use_blas_only =
        false; ///< Whether to use BLAS-only computation (no PDX pruning) for all iterations

    // Convergence parameters
    float tol = 1e-8f;              ///< Tolerance for shift-based early termination
    float recall_tol = 0.001f;      ///< Tolerance for recall-based early termination
    bool early_termination = false; ///< Whether to stop early on convergence

    // Query/Recall parameters
    bool sample_queries = false;        ///< Whether to sample queries from data
    size_t objective_k = 100;           ///< Number of nearest neighbors for recall computation
    float ann_explore_fraction = 0.01f; ///< Fraction of centroids to explore (0.0 to 1.0)

    // Pruning parameters
    float min_not_pruned_pct = 0.03f; ///< Minimum percentage of vectors not pruned (3% = 97% pruned)
    float max_not_pruned_pct = 0.05f; ///< Maximum percentage of vectors not pruned (5% = 95% pruned)
    float adjustment_factor_for_partial_d = 0.20f; ///< Adjustment factor for partial_d tuning (20%)

    // Output parameters
    bool unrotate_centroids = true;   ///< Whether to unrotate centroids before returning
    bool perform_assignments = false; ///< Whether to perform final assignment pass
    bool verbose = false;             ///< Whether to print progress information
    bool angular = false;
};

/**
 * @brief Statistics for a single iteration of SuperKMeans clustering.
 */
struct SuperKMeansIterationStats {
    size_t iteration = 0;   ///< Iteration number (1-indexed)
    float objective = 0.0f; ///< Total clustering cost (sum of distances)
    float shift = 0.0f;     ///< Average squared centroid shift from previous iteration
    size_t split = 0;       ///< Number of clusters that were split (empty cluster handling)
    float recall = 0.0f;    ///< Recall@k value (0.0 to 1.0, only when queries provided)
    float not_pruned_pct =
        -1.0f; ///< Percentage of vectors not pruned (0.0 to 1.0, -1.0 if not applicable)
    uint32_t partial_d =
        0; ///< Number of dimensions used for partial distance computation (0 if not applicable)
    bool is_blas_only =
        false; ///< Whether this iteration used BLAS-only computation (no PDX pruning)
};

template <Quantization q = Quantization::f32, DistanceFunction alpha = DistanceFunction::l2>
class SuperKMeans {
    using centroid_value_t = skmeans_centroid_value_t<q>;
    using vector_value_t = skmeans_value_t<q>;
    using Pruner = ADSamplingPruner<q>;
    using layout_t = PDXLayout<q, alpha>;
    using distance_t = skmeans_distance_t<q>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;
    using batch_computer = gpu::BatchComputer<alpha, q>;
    using cpu_batch_computer = BatchComputer<alpha, q>; // This batch computer is always CPU based, as GPU does not implement FindKNearestNeighbors

    static constexpr size_t RECALL_CONVERGENCE_PATIENCE = 2;

  public:
    /**
     * @brief Constructs a SuperKMeans instance with custom configuration.
     *
     * @param n_clusters Number of clusters to create
     * @param dimensionality Number of dimensions in the data
     * @param config Configuration parameters (see SuperKMeansConfig)
     */
    SuperKMeans(size_t n_clusters, size_t dimensionality, const SuperKMeansConfig& config)
        : _d(dimensionality), _n_clusters(n_clusters), _config(config) {
        SKMEANS_ENSURE_POSITIVE(n_clusters);
        SKMEANS_ENSURE_POSITIVE(dimensionality);
        SKMEANS_ENSURE_POSITIVE(_config.iters);
        SKMEANS_ENSURE_POSITIVE(_config.sampling_fraction);
        if (_config.sampling_fraction > 1.0) {
            throw std::invalid_argument("sampling_fraction must be <= 1.0");
        }
        // Set thread count: 0 means auto-detect max available
        _n_threads = (_config.n_threads == 0) ? omp_get_max_threads() : _config.n_threads;
        g_n_threads = _n_threads; // Also set global for external functions
        _pruner = std::make_unique<Pruner>(dimensionality, PRUNER_INITIAL_THRESHOLD, _config.seed);
    }

    /**
     * @brief Constructs a SuperKMeans instance with default configuration.
     *
     * @param n_clusters Number of clusters to create
     * @param dimensionality Number of dimensions in the data
     */
    SuperKMeans(size_t n_clusters, size_t dimensionality)
        : SuperKMeans(n_clusters, dimensionality, SuperKMeansConfig{}) {}

    /**
     * @brief Performs the clustering on the provided data.
     *
     * @param data Pointer to the data matrix (row-major, n × d)
     * @param n Number of points (rows) in the data matrix
     * @param queries Optional pointer to query vectors for recall computation
     * @param n_queries Number of query vectors (ignored if queries is nullptr and sample_queries is
     * false)
     * @return std::vector<skmeans_centroid_value_t<q>> Trained centroids
     */
    std::vector<skmeans_centroid_value_t<q>> Train(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n,
        const vector_value_t* SKM_RESTRICT queries = nullptr,
        const size_t n_queries = 0
    ) {
        SKMEANS_ENSURE_POSITIVE(n);
        if (_trained) {
            throw std::runtime_error("The clustering has already been trained");
        }

        iteration_stats.clear();

        if (n < _n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters"
            );
        }
        if (n_queries > 0 && queries == nullptr && !_config.sample_queries) {
            throw std::invalid_argument(
                "Queries must be provided if n_queries > 0 and sample_queries is false"
            );
        }
        const vector_value_t* SKM_RESTRICT data_p = data;
        _n_samples = GetNVectorsToSample(n);

        {
            SKM_PROFILE_SCOPE("allocator");
            _centroids.resize(_n_clusters * _d);
            _horizontal_centroids.resize(_n_clusters * _d);
            _prev_centroids.resize(_n_clusters * _d);
            _cluster_sizes.resize(_n_clusters);
            _assignments.resize(n);
            _distances.resize(n);
            _data_norms.resize(_n_samples);
            _centroid_norms.resize(_n_clusters);
        }
        std::vector<distance_t> all_distances(X_BATCH_SIZE * Y_BATCH_SIZE);
        if (_config.verbose) {
            std::cout << "Generating centroids..." << std::endl;
        }
        //! _centroids and _partial_horizontal_centroids are always wrapped with the PDXLayout
        //! object
        _vertical_d = PDXLayout<q, alpha>::GetDimensionSplit(_d).vertical_d;
        // Set initial_partial_d dynamically as half of vertical_d
        _initial_partial_d = std::max<uint32_t>(MIN_PARTIAL_D, _vertical_d / 2);
        // Ensure initial_partial_d doesn't exceed vertical_d to avoid double-counting dimensions
        // when BLAS computes more dimensions than the vertical block contains
        if (_initial_partial_d > _vertical_d) {
            _initial_partial_d = _vertical_d;
        }
        if (_config.verbose) {
            std::cout << "Vertical D: " << _vertical_d << std::endl;
            std::cout << "Horizontal D: " << _d - _vertical_d << std::endl;
            std::cout << "Initial Partial D: " << _initial_partial_d << std::endl;
        }
        {
            SKM_PROFILE_SCOPE("allocator");
            _partial_horizontal_centroids.resize(_n_clusters * _vertical_d);
        }
        auto centroids_pdx_wrapper = GenerateCentroids(data_p);
        if (_config.verbose) {
            std::cout << "Sampling data..." << std::endl;
        }

        std::vector<vector_value_t> data_samples_buffer;
        // TODO(@lkuffo, crit): If I rotate the vectors, and then sample. 
        //   I don't need to rotate the sampled vectors again.
        SampleVectors(data_p, data_samples_buffer, n, _n_samples);
        auto data_to_cluster = data_samples_buffer.data();

        // We use _prev_centroids to store the rotated initial centroids, to avoid allocating a new
        // buffer
        {
            SKM_PROFILE_SCOPE("rotator");
            if (_config.verbose)
                std::cout << "Rotating..." << std::endl;
            _pruner->Rotate(_horizontal_centroids.data(), _prev_centroids.data(), _n_clusters);
        }

        // Getting norms for data and initial centroids (full norms)
        GetL2NormsRowMajor(data_to_cluster, _n_samples, _data_norms.data());
        GetL2NormsRowMajor(_prev_centroids.data(), _n_clusters, _centroid_norms.data());

        std::vector<vector_value_t> rotated_queries;
        if (n_queries) {
            // Compute number of centroids to explore from the fraction parameter (minimum 1)
            _centroids_to_explore = std::max<size_t>(
                static_cast<size_t>(_n_clusters * _config.ann_explore_fraction), 1
            );
            if (_config.verbose) {
                std::cout << "Centroids to explore: " << _centroids_to_explore << " ("
                          << _config.ann_explore_fraction * 100.0f << "% of " << _n_clusters << ")"
                          << std::endl;
            }
            {
                SKM_PROFILE_SCOPE("allocator");
                _gt_assignments.resize(n_queries * _config.objective_k);
                _gt_distances.resize(n_queries * _config.objective_k);
            }
            // Create temporary buffer for rotated queries and rotate them
            rotated_queries.resize(n_queries * _d);
            if (_config.sample_queries) {
                std::cout << "Sampling queries from data..." << std::endl;
                SampleVectors<false>(
                    data_to_cluster, rotated_queries, _n_samples, n_queries
                );
            } else {
                // We already did a validation step to ensure that queries is not nullptr
                SKM_PROFILE_SCOPE("rotator");
                _pruner->Rotate(queries, rotated_queries.data(), n_queries);
            }
            // Compute and cache query norms once (used by ComputeRecall in each iteration)
            _query_norms.resize(n_queries);
            GetL2NormsRowMajor(rotated_queries.data(), n_queries, _query_norms.data());
            GetGTAssignmentsAndDistances(data_to_cluster, rotated_queries.data(), n_queries);
        }

        // First iteration: Only Blas
        InitAssignAndUpdateCentroids(data_to_cluster, _prev_centroids.data(), all_distances.data());
        ConsolidateCentroids();
        ComputeCost();
        ComputeShift();
        if (n_queries) {
            _recall = ComputeRecall(rotated_queries.data(), n_queries);
        }
        size_t iter_idx = 1;
        {
            SuperKMeansIterationStats stats;
            stats.iteration = 1;
            stats.objective = _cost;
            stats.shift = _shift;
            stats.split = _n_split;
            stats.recall = _recall;
            stats.is_blas_only = false;
            iteration_stats.push_back(stats);
        }
        if (_config.verbose)
            std::cout << "Iteration 1" << "/" << _config.iters << " | Objective: " << _cost
                      << " | Shift: " << _shift << " | Split: " << _n_split
                      << " | Recall: " << _recall << std::endl
                      << std::endl;
        // End of First iteration

        if (_config.iters <= 1) {
            auto output_centroids = GetOutputCentroids(_config.unrotate_centroids);
            if (_config.perform_assignments) {
                _assignments = Assign(data, output_centroids.data(), n, _n_clusters);
            }
            Profiler::Get().PrintHierarchical();
            return output_centroids;
        }

        // Early stopping tracking variables
        float best_recall = _recall;
        size_t iters_without_improvement = 0;

        // Special path for low-dimensional data or when use_blas_only is enabled: use BLAS-only for
        // all iterations
        if (_d < 128 || _config.use_blas_only || _n_clusters <= 128) {
            for (; iter_idx < _config.iters; ++iter_idx) {
                // After swap: _prev_centroids has old centroids, _horizontal_centroids will be
                // zeroed for accumulation
                std::swap(_horizontal_centroids, _prev_centroids);
                // Recompute centroid norms for the (now in _prev_centroids) old centroids
                GetL2NormsRowMajor(_prev_centroids.data(), _n_clusters, _centroid_norms.data());
                InitAssignAndUpdateCentroids(
                    data_to_cluster,
                    _prev_centroids.data(), // Search using old centroids
                    all_distances.data()
                );
                ConsolidateCentroids();
                ComputeCost();
                ComputeShift();
                if (n_queries) {
                    // Update centroid norms to match the NEW centroids (after ConsolidateCentroids)
                    GetL2NormsRowMajor(
                        _horizontal_centroids.data(), _n_clusters, _centroid_norms.data()
                    );
                    _recall = ComputeRecall(rotated_queries.data(), n_queries);
                }
                {
                    SuperKMeansIterationStats stats;
                    stats.iteration = iter_idx + 1;
                    stats.objective = _cost;
                    stats.shift = _shift;
                    stats.split = _n_split;
                    stats.recall = _recall;
                    stats.is_blas_only = true;
                    iteration_stats.push_back(stats);
                }
                if (_config.verbose)
                    std::cout << "Iteration " << iter_idx + 1 << "/" << _config.iters
                              << " | Objective: " << _cost << " | Shift: " << _shift
                              << " | Split: " << _n_split << " | Recall: " << _recall
                              << " [BLAS-only]" << std::endl
                              << std::endl;
                // Early stopping if converged
                if (_config.early_termination &&
                    ShouldStopEarly(
                        n_queries > 0, best_recall, iters_without_improvement, iter_idx
                    )) {
                    break;
                }
            }
            _trained = true;
            auto output_centroids = GetOutputCentroids(_config.unrotate_centroids);
            if (_config.perform_assignments) {
                _assignments = Assign(data, output_centroids.data(), n, _n_clusters);
            }
            if (_config.verbose) {
                Profiler::Get().PrintHierarchical();
            }
            return output_centroids;
        }

        // Rest of iterations
        std::vector<vector_value_t> centroids_partial_norms(_n_clusters);
        // Buffer to store per-vector not-pruned counts for tuning _initial_partial_d
        std::vector<size_t> not_pruned_counts(_n_samples);
        GetPartialL2NormsRowMajor(data_to_cluster, _n_samples, _data_norms.data());

				gpu::GPUDeviceContext<skmeans_value_t<q>, skmeans_value_t<q>> gpu_device_context(_n_samples, _n_clusters, _d, GPU_STREAM_POOL_SIZE);
				gpu_device_context.x.copy_to_device(data_to_cluster);
        for (; iter_idx < _config.iters; ++iter_idx) {
            // After swap: _prev_centroids has old centroids, _horizontal_centroids will be zeroed
            // for accumulation
            std::swap(_horizontal_centroids, _prev_centroids);
            GetL2NormsRowMajor(
                _prev_centroids.data(),
                _n_clusters,
                centroids_partial_norms.data(),
                _initial_partial_d
            );
            // Reset the not-pruned counts buffer
            std::fill(not_pruned_counts.begin(), not_pruned_counts.end(), 0);
            AssignAndUpdateCentroidsPartialBatched(
								gpu_device_context,
                data_to_cluster,
                _prev_centroids.data(), // Search using _prev_centroids
                centroids_partial_norms.data(),
                all_distances.data(),
                centroids_pdx_wrapper,
                not_pruned_counts.data()
            );

            // Tune _initial_partial_d based on the average not-pruned percentage
            bool partial_d_changed = false;
            float avg_not_pruned_pct = TuneInitialPartialD(
                not_pruned_counts.data(), _n_samples, _n_clusters, partial_d_changed
            );
            // If _initial_partial_d changed, recompute the data norms with the new partial_d
            if (partial_d_changed) {
                GetPartialL2NormsRowMajor(data_to_cluster, _n_samples, _data_norms.data());
            }

            ConsolidateCentroids();
            ComputeCost();
            ComputeShift();
            if (n_queries) {
                // Update centroid norms with FULL norms for recall computation
                // (PDX uses partial norms for distance computation, but recall needs full norms)
                GetL2NormsRowMajor(
                    _horizontal_centroids.data(), _n_clusters, _centroid_norms.data()
                );
                _recall = ComputeRecall(rotated_queries.data(), n_queries);
            }
            {
                SuperKMeansIterationStats stats;
                stats.iteration = iter_idx + 1;
                stats.objective = _cost;
                stats.shift = _shift;
                stats.split = _n_split;
                stats.recall = _recall;
                stats.not_pruned_pct = avg_not_pruned_pct;
                stats.partial_d = _initial_partial_d;
                stats.is_blas_only = false;
                iteration_stats.push_back(stats);
            }
            if (_config.verbose)
                std::cout << "Iteration " << iter_idx + 1 << "/" << _config.iters
                          << " | Objective: " << _cost << " | Shift: " << _shift
                          << " | Split: " << _n_split << " | Recall: " << _recall
                          << " | Not Pruned %: " << avg_not_pruned_pct * 100.0f
                          << " | Partial D: " << _initial_partial_d << std::endl
                          << std::endl;
            // Early stopping if converged
            if (_config.early_termination &&
                ShouldStopEarly(n_queries > 0, best_recall, iters_without_improvement, iter_idx)) {
                break;
            }
        }
        // Note: When sampling_fraction < 1, only the first n_samples vectors have assignments.
        // Users can call Assign() on remaining vectors if needed.

        _trained = true;
        auto output_centroids = GetOutputCentroids(_config.unrotate_centroids);
        if (_config.perform_assignments) {
            _assignments = Assign(data, output_centroids.data(), n, _n_clusters);
        }
        if (_config.verbose) {
            Profiler::Get().PrintHierarchical();
        }
        return output_centroids;
    }

    /**
     * @brief Assign vectors to their nearest centroid using BLAS-based computation.
     *
     * Both vectors and centroids are assumed to be in the same domain (no rotation/transformation
     * needed).
     *
     * @param vectors The data matrix (row-major, n_vectors x d)
     * @param centroids The centroids matrix (row-major, n_centroids x d)
     * @param n_vectors Number of vectors
     * @param n_centroids Number of centroids
     * @param d Dimensionality of vectors and centroids
     * @return std::vector<uint32_t> Assignment for each vector (index of nearest centroid)
     */
    [[nodiscard]] std::vector<uint32_t> Assign(
        const vector_value_t* SKM_RESTRICT vectors,
        const vector_value_t* SKM_RESTRICT centroids,
        const size_t n_vectors,
        const size_t n_centroids
    ) {
        SKM_PROFILE_SCOPE("assign");

        // Compute norms for vectors and centroids
        std::vector<vector_value_t> vector_norms(n_vectors);
        std::vector<vector_value_t> centroid_norms_local(n_centroids);
        {
            Eigen::Map<const MatrixR> vectors_mat(vectors, n_vectors, _d);
            Eigen::Map<VectorR> v_norms(vector_norms.data(), n_vectors);
            v_norms.noalias() = vectors_mat.rowwise().squaredNorm();
        }
        {
            Eigen::Map<const MatrixR> centroids_mat(centroids, n_centroids, _d);
            Eigen::Map<VectorR> c_norms(centroid_norms_local.data(), n_centroids);
            c_norms.noalias() = centroids_mat.rowwise().squaredNorm();
        }

        // Allocate output and temporary buffers
        std::vector<uint32_t> assignments(n_vectors);
        std::vector<distance_t> distances(n_vectors);
        std::vector<distance_t> all_distances_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

        // Use batched BLAS computation for assignment
        batch_computer::FindNearestNeighbor(
            vectors,
            centroids,
            n_vectors,
            n_centroids,
            _d,
            vector_norms.data(),
            centroid_norms_local.data(),
            assignments.data(),
            distances.data(),
            all_distances_buf.data()
        );

        return assignments;
    }

    /** @brief Returns the number of clusters. */
    [[nodiscard]] inline size_t GetNClusters() const noexcept { return _n_clusters; }

    /** @brief Returns whether the model has been trained. */
    [[nodiscard]] inline bool IsTrained() const noexcept { return _trained; }

  protected:
    /**
     * @brief Performs initial assignment and centroid update using BLAS-based computation.
     *
     * Used for the first iteration where full distance computation via BLAS is used
     * (no pruning). Assigns each data point to its nearest centroid, then updates
     * centroid positions.
     *
     * @param data Data matrix (row-major, _n_samples × _d)
     * @param rotated_initial_centroids Initial centroids (row-major, _n_clusters × _d)
     * @param all_distances Workspace buffer for distance computations
     */
    void InitAssignAndUpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT rotated_initial_centroids,
        distance_t* SKM_RESTRICT all_distances
    ) {
        batch_computer::FindNearestNeighbor(
            data,
            rotated_initial_centroids,
            _n_samples,
            _n_clusters,
            _d,
            _data_norms.data(),
            _centroid_norms.data(),
            _assignments.data(),
            _distances.data(),
            all_distances
        );
        std::fill(_horizontal_centroids.begin(), _horizontal_centroids.end(), 0.0);
        std::fill(_cluster_sizes.begin(), _cluster_sizes.end(), 0);
        _cost = 0.0;
        UpdateCentroids(data);
    }

    /**
     * @brief Performs assignment and centroid update using hybrid BLAS+PDX computation.
     *
     * Uses BLAS for partial distance computation (first _initial_partial_d dimensions),
     * then PDXearch for pruning and completing distances for remaining candidates.
     *
     * @param data Data matrix (row-major, _n_samples × _d)
     * @param centroids_for_search Centroids to use for BLAS distance computation (row-major)
     * @param partial_centroid_norms Partial norms of centroids (first _initial_partial_d dims)
     * @param all_distances Workspace buffer for distance computations
     * @param pdx_centroids PDX-layout centroids for PDXearch
     * @param out_not_pruned_counts Optional output for per-vector pruning statistics
     */
    void AssignAndUpdateCentroidsPartialBatched(
				gpu::GPUDeviceContext<skmeans_value_t<q>, skmeans_value_t<q>>& gpu_device_context,
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT centroids_for_search,
        const vector_value_t* SKM_RESTRICT partial_centroid_norms,
        distance_t* SKM_RESTRICT all_distances,
        const layout_t& pdx_centroids,
        size_t* out_not_pruned_counts = nullptr
    ) {
        _cost = 0.0;
        batch_computer::FindNearestNeighborWithPruning(
						gpu_device_context,
            data,
            centroids_for_search,
            _n_samples,
            _n_clusters,
            _d,
            _data_norms.data(),
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
        UpdateCentroids(data);
    }

    /**
     * @brief Updates centroid positions by accumulating assigned vectors.
     *
     * After this call, _horizontal_centroids contains the sum of assigned vectors.
     * ConsolidateCentroids() must be called to normalize by cluster sizes.
     *
     * @param data Data matrix (row-major, _n_samples × _d)
     */
    void UpdateCentroids(const vector_value_t* SKM_RESTRICT data) {
        SKM_PROFILE_SCOPE("update_centroids");
#pragma omp parallel num_threads(_n_threads)
        {
            uint32_t nt = _n_threads;
            uint32_t rank = omp_get_thread_num();
            // This thread is taking care of centroids c0:c1
            size_t c0 = (_n_clusters * rank) / nt;
            size_t c1 = (_n_clusters * (rank + 1)) / nt;
            for (size_t i = 0; i < _n_samples; i++) {
                uint32_t ci = _assignments[i];
                assert(ci >= 0 && ci < _n_clusters);
                if (ci >= c0 && ci < c1) {
                    auto vector_p = data + i * _d;
                    _cluster_sizes[ci] += 1;
                    UpdateCentroid(vector_p, ci);
                }
            }
        }
    }

    /**
     * @brief Adds a vector to its assigned centroid's accumulator.
     */
    SKM_ALWAYS_INLINE void UpdateCentroid(
        const vector_value_t* SKM_RESTRICT vector,
        const uint32_t cluster_idx
    ) {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < _d; ++i) {
            _horizontal_centroids[cluster_idx * _d + i] += vector[i];
        }
    }

    /**
     * @brief Handles empty clusters by splitting large clusters.
     * Taken from Faiss implementation:
     * https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.cpp
     *
     * When a cluster becomes empty (no points assigned), this method splits
     * a large cluster to repopulate it. Selection is probabilistic based on
     * cluster sizes.
     */
    void SplitClusters() {
        _n_split = 0;
        std::default_random_engine rng(_config.seed);
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

    /**
     * @brief Finalizes centroid computation after assignment.
     *
     * Divides accumulated sums by cluster sizes to get mean centroids,
     * handles empty clusters via splitting, and converts to PDX layout.
     */
    void ConsolidateCentroids() {
        SKM_PROFILE_SCOPE("consolidate");
        {
            SKM_PROFILE_SCOPE("consolidate/splitting");
#pragma omp parallel for if (_n_threads > 1) num_threads(_n_threads)
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
            SKM_PROFILE_SCOPE("consolidate/normalize");
            if (_config.angular) {
                PostprocessCentroids();
            }
        }
        {
            SKM_PROFILE_SCOPE("consolidate/pdxify");
            //! This updates the object within the pdx_layout wrapper
            PDXLayout<q, alpha>::template PDXify<false>(
                _horizontal_centroids.data(), _centroids.data(), _n_clusters, _d
            );
            CentroidsToAuxiliaryHorizontal();
        }
    }

    /**
     * @brief Computes the total clustering cost (sum of distances).
     */
    void ComputeCost() {
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < _n_samples; ++i) {
            _cost += _distances[i];
        }
    }

    /**
     * @brief Computes the average squared centroid shift from previous iteration.
     *
     * Used for convergence detection - small shift indicates centroids have stabilized.
     */
    void ComputeShift() {
        SKM_PROFILE_SCOPE("shift");
        Eigen::Map<const MatrixR> new_mat(_horizontal_centroids.data(), _n_clusters, _d);
        Eigen::Map<const MatrixR> prev_mat(_prev_centroids.data(), _n_clusters, _d);
        _shift = 0.0f;
#pragma omp parallel for reduction(+ : _shift) num_threads(_n_threads)
        for (size_t i = 0; i < _n_clusters; ++i) {
            _shift += (new_mat.row(i) - prev_mat.row(i)).squaredNorm();
        }
        _shift /= (_n_clusters * _d);
    }

    /**
     * @brief Computes ground truth assignments for recall calculation.
     *
     * Finds the top-k nearest data points for each query using exact search.
     * These assignments are used as ground truth for evaluating centroid quality.
     *
     * @param data Data matrix (sampled data points)
     * @param queries Query vectors
     * @param n_queries Number of query vectors
     */
    void GetGTAssignmentsAndDistances(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT queries,
        const size_t n_queries
    ) {
        SKM_PROFILE_SCOPE("gt_assignments");
        _tmp_distances_buffer.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        std::vector<distance_t> query_norms(n_queries);
        GetL2NormsRowMajor(queries, n_queries, query_norms.data());
				cpu_batch_computer::FindKNearestNeighbors(
            queries,
            data,
            n_queries,
            _n_samples,
            _d,
            query_norms.data(),
            _data_norms.data(),
            _config.objective_k,
            _gt_assignments.data(),
            _gt_distances.data(),
            _tmp_distances_buffer.data()
        );
    }

    /**
     * @brief Computes recall@k for current centroids.
     *
     * For each query, checks how many of its ground truth nearest neighbors
     * would be found when searching only the top centroids. Higher recall
     * indicates better centroid quality for ANN indexing.
     *
     * @param queries Query vectors
     * @param n_queries Number of query vectors
     * @return Recall value (0.0 to 1.0)
     */
    float ComputeRecall(const vector_value_t* SKM_RESTRICT queries, const size_t n_queries) {
        SKM_PROFILE_SCOPE("recall");
        _tmp_distances_buffer.resize(X_BATCH_SIZE * Y_BATCH_SIZE);
        _promising_centroids.resize(n_queries * _centroids_to_explore);
        _recall_distances.resize(n_queries * _centroids_to_explore);
        cpu_batch_computer::FindKNearestNeighbors(
            queries,
            _horizontal_centroids.data(),
            n_queries,
            _n_clusters,
            _d,
            _query_norms.data(),
            _centroid_norms.data(),
            _centroids_to_explore,
            _promising_centroids.data(),
            _recall_distances.data(),
            _tmp_distances_buffer.data()
        );
        // For each query, compute recall@objective_k: how many of the GT clusters are found in the
        // top-64 assignments Recall per query = (# matched GT assignments in top-64) / objective_k
        // Final recall = average over all queries
        float sum_recall = 0.0f;
        for (size_t i = 0; i < n_queries; ++i) {
            size_t found_in_query = 0;
            // For each GT assignment for query q
            for (size_t j = 0; j < _config.objective_k; ++j) {
                uint32_t gt = _gt_assignments[i * _config.objective_k + j]; // gt is a vector index
                // Check if this GT assignment is present in the top-64 assignments for this query
                bool found = false;
                for (size_t t = 0; t < _centroids_to_explore; ++t) {
                    // If a promising centroid is the same as the GT centroid assignment, then we
                    // have a match
                    if (_promising_centroids[i * _centroids_to_explore + t] == _assignments[gt]) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    ++found_in_query;
                }
            }
            sum_recall +=
                static_cast<float>(found_in_query) / static_cast<float>(_config.objective_k);
        }
        return sum_recall / static_cast<float>(n_queries);
    }

    /**
     * @brief Generates initial centroids from the data.
     *
     * Takes the first _n_clusters vectors as initial centroids (similar to FAISS),
     * rotates them, converts to PDX layout, and creates a PDXLayout wrapper.
     *
     * @param data Data matrix
     * @param n Number of data points
     * @return PDXLayout wrapper for the centroids
     */
    PDXLayout<q, alpha> GenerateCentroids(const vector_value_t* SKM_RESTRICT data) {
        {
            SKM_PROFILE_SCOPE("sampling");
            auto tmp_centroids_p = _horizontal_centroids.data();

            // Random sampling without replacement
            std::mt19937 rng(_config.seed);
            std::vector<size_t> indices(_n_samples);
            for (size_t i = 0; i < _n_samples; ++i) {
                indices[i] = i;
            }
            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t i = 0; i < _n_clusters; i += 1) {
                memcpy(
                    (void*) tmp_centroids_p,
                    (void*) (data + (indices[i] * _d)),
                    sizeof(centroid_value_t) * _d
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
            PDXLayout<q, alpha>::template PDXify<false>(
                rotated_centroids.data(), _centroids.data(), _n_clusters, _d
            );
        }
        //! We wrap _centroids and _partial_horizontal_centroids in the PDXLayout wrapper
        //! Any updates to these objects is reflected in the PDXLayout
        auto pdx_centroids = PDXLayout<q, alpha>(
            _centroids.data(), *_pruner, _n_clusters, _d, _partial_horizontal_centroids.data()
        );
        return pdx_centroids;
    }

    /**
     * @brief Computes partial L2 squared norms (first _initial_partial_d dimensions).
     */
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

    /**
     * @brief Computes full L2 squared norms for each vector.
     */
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

    /**
     * @brief Computes partial L2 squared norms (first partial_d dimensions).
     * @param partial_d Number of dimensions to include in norm computation
     */
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
     * @brief Copies the first _vertical_d dimensions of centroids for efficient pruning on
     * PDXearch.
     * TODO(@lkuffo, high): We can avoid _partial_horizontal_centroids by using the full horizontal
     * centroids in PDXearch.
     */
    void CentroidsToAuxiliaryHorizontal() {
        Eigen::Map<MatrixR> hor_centroids(_horizontal_centroids.data(), _n_clusters, _d);
        Eigen::Map<MatrixR> out_aux_centroids(
            _partial_horizontal_centroids.data(), _n_clusters, _vertical_d
        );
        out_aux_centroids.noalias() = hor_centroids.leftCols(_vertical_d);
    }

    /**
     * @brief Tune _initial_partial_d based on the average not-pruned percentage.
     *
     * A safe range for pruning is between 75% - 90% of vectors pruned (i.e., 10% - 25% not pruned).
     * - If avg_not_pruned_pct > 25% (i.e., less than 75% pruned), we reduce _initial_partial_d by
     * 25% to be more aggressive in pruning
     * - If avg_not_pruned_pct < 10% (i.e., more than 90% pruned), we increase _initial_partial_d by
     * 25% to be less aggressive
     * - _initial_partial_d is clamped between 8 and _vertical_d
     *
     * @param not_pruned_counts Buffer containing per-vector not-pruned counts
     * @param n_samples Number of X vectors
     * @param n_y Number of Y vectors (centroids)
     * @param partial_d_changed Output parameter: set to true if _initial_partial_d was changed
     * @return The computed average not-pruned percentage
     */
    float TuneInitialPartialD(
        const size_t* not_pruned_counts,
        size_t n_samples,
        size_t n_y,
        bool& partial_d_changed
    ) {

        // Calculate average not-pruned percentage from the buffer
        float avg_not_pruned_pct = 0.0f;
        for (size_t i = 0; i < n_samples; ++i) {
            avg_not_pruned_pct += static_cast<float>(not_pruned_counts[i]);
        }
        avg_not_pruned_pct /= static_cast<float>(n_samples * n_y);

        uint32_t old_partial_d = _initial_partial_d;

        if (avg_not_pruned_pct > _config.max_not_pruned_pct) {
            // Too many vectors not pruned (< max_not_pruned_pct pruned), need more BLAS dimensions
            // Increase _initial_partial_d by adjustment_factor_for_partial_d
            // When we increase we have to be more aggresive
            uint32_t increase = static_cast<uint32_t>(_initial_partial_d * _config.adjustment_factor_for_partial_d * 2);
            _initial_partial_d = std::min(_initial_partial_d + std::max(increase, 1u), _vertical_d);
        } else if (avg_not_pruned_pct < _config.min_not_pruned_pct) {
            // Too few vectors not pruned (> min_not_pruned_pct pruned), can reduce BLAS dimensions
            // Decrease _initial_partial_d by adjustment_factor_for_partial_d
            uint32_t decrease = static_cast<uint32_t>(_initial_partial_d * _config.adjustment_factor_for_partial_d);
            _initial_partial_d =
                std::max(_initial_partial_d - std::max(decrease, 1u), MIN_PARTIAL_D);
        }
        partial_d_changed = (old_partial_d != _initial_partial_d);

        // else: within safe range (75% - 90% pruned), no adjustment needed
        if (_config.verbose && partial_d_changed) {
            std::cout << "Tuning _initial_partial_d: " << old_partial_d << " -> "
                      << _initial_partial_d << " (avg not pruned: " << avg_not_pruned_pct * 100.0f
                      << "%)" << std::endl;
        }

        return avg_not_pruned_pct;
    }

    /**
     * @brief Computes the number of vectors to sample based on sampling_fraction.
     * @param n Total number of vectors
     * @return Number of vectors to sample
     */
    size_t GetNVectorsToSample(const size_t n) const {
        if (_config.sampling_fraction == 1.0) {
            return n;
        }
        return std::floor(n * _config.sampling_fraction);
    }

    /**
     * @brief Check if training should stop early based on convergence criteria.
     *
     * Convergence is detected when either:
     * - Shift is below tolerance (_shift < _config.tol)
     * - Recall hasn't improved by more than _config.recall_tol in RECALL_CONVERGENCE_PATIENCE
     * consecutive iterations (when tracking recall)
     *
     * @param tracking_recall Whether recall is being tracked (n_queries > 0)
     * @param best_recall Reference to the best recall seen so far (updated if current is better)
     * @param iters_without_improvement Reference to counter of iterations without recall
     * improvement
     * @param iter_idx Current iteration index (for verbose output)
     * @return true if training should stop, false otherwise
     */
    bool ShouldStopEarly(
        bool tracking_recall,
        float& best_recall,
        size_t& iters_without_improvement,
        size_t iter_idx
    ) {
        // Check shift convergence
        if (_shift < _config.tol) {
            if (_config.verbose)
                std::cout << "Converged at iteration " << iter_idx + 1 << " (shift " << _shift
                          << " < tol " << _config.tol << ")" << std::endl;
            return true;
        }

        // Check recall convergence (only when tracking recall)
        if (tracking_recall) {
            float improvement = _recall - best_recall;
            if (improvement > _config.recall_tol) {
                // Significant improvement: update best and reset counter
                best_recall = _recall;
                iters_without_improvement = 0;
            } else {
                // No significant improvement
                iters_without_improvement++;
                if (iters_without_improvement >= RECALL_CONVERGENCE_PATIENCE) {
                    if (_config.verbose)
                        std::cout << "Converged at iteration " << iter_idx + 1 << " (recall "
                                  << _recall << " hasn't improved by more than "
                                  << _config.recall_tol << " in " << RECALL_CONVERGENCE_PATIENCE
                                  << " iterations, best: " << best_recall << ")" << std::endl;
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * @brief Prepare centroids for output (applies any necessary transformations).
     * @param should_unrotate If true, unrotates centroids to original space; if false, returns
     * rotated centroids.
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

    /**
     * @brief Normalizes centroids to unit length for dot product distance.
     *
     * Called when using DistanceFunction::dp to ensure centroids are on the unit sphere.
     */
    void PostprocessCentroids() {
        auto horizontal_centroids_p = _horizontal_centroids.data();
        for (size_t i = 0; i < _n_clusters; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < _d; ++j) {
                sum += horizontal_centroids_p[i * _d + j] * horizontal_centroids_p[i * _d + j];
            }
            float norm = 1.0f / std::sqrt(sum);
            for (size_t j = 0; j < _d; ++j) {
                horizontal_centroids_p[i * _d + j] *= norm;
            }
        }
    }

    /**
     * @brief Samples and optionally rotates vectors for training.
     *
     * Performs random sampling without replacement using shuffle technique,
     * and optionally rotates the sampled vectors using the pruner's rotation matrix.
     *
     * @tparam ROTATE Whether to apply rotation (default true)
     * @param data Input data matrix
     * @param out_buffer Output buffer for sampled (and optionally rotated) vectors
     * @param n Total number of input vectors
     * @param n_samples Number of vectors to sample
     */
    template <bool ROTATE = true>
    void SampleVectors(
        const vector_value_t* SKM_RESTRICT data,
        std::vector<vector_value_t>& out_buffer,
        const size_t n,
        const size_t n_samples
    ) {
        out_buffer.resize(n_samples * _d);
        if (_config.verbose)
            std::cout << "n_samples: " << n_samples << std::endl;

        // Intermediate buffer needed only when both sampling and rotating
        // (we have not yet implemented rotation in-place)
        std::vector<vector_value_t> samples_tmp;
        const vector_value_t* src_data = data;

        if (n_samples < n) {
            SKM_PROFILE_SCOPE("sampling");
            // Random sampling without replacement using shuffle
            std::mt19937 rng(_config.seed);
            std::vector<size_t> indices(n);
            for (size_t i = 0; i < n; ++i) {
                indices[i] = i;
            }
            std::shuffle(indices.begin(), indices.end(), rng);

            if constexpr (ROTATE) {
                // Need intermediate buffer: sample first, then rotate
                samples_tmp.resize(n_samples * _d);
                for (size_t i = 0; i < n_samples; ++i) {
                    memcpy(
                        (void*) (samples_tmp.data() + i * _d),
                        (void*) (data + indices[i] * _d),
                        sizeof(vector_value_t) * _d
                    );
                }
                src_data = samples_tmp.data();
            } else {
                // No rotation: copy directly into output buffer
                for (size_t i = 0; i < n_samples; ++i) {
                    memcpy(
                        (void*) (out_buffer.data() + i * _d),
                        (void*) (data + indices[i] * _d),
                        sizeof(vector_value_t) * _d
                    );
                }
                return; // Done, no rotation needed
            }
        }

        // Rotate or copy into output buffer
        SKM_PROFILE_SCOPE("rotator");
        if constexpr (ROTATE) {
            _pruner->Rotate(src_data, out_buffer.data(), n_samples);
        } else {
            memcpy(
                (void*) out_buffer.data(), (void*) src_data, sizeof(vector_value_t) * n_samples * _d
            );
        }
    }

    // === Configuration ===
    const size_t _d;
    const size_t _n_clusters;
    SuperKMeansConfig _config;
    uint32_t _n_threads;

    // === Training state ===
    bool _trained = false;
    size_t _n_samples = 0;
    size_t _n_split = 0;
    size_t _centroids_to_explore = 0;
    uint32_t _vertical_d = 0;
    uint32_t _initial_partial_d = 0;
    float _cost = 0.0f;
    float _shift = 0.0f;
    float _recall = 0.0f;

    // === Core algorithm components ===
    std::unique_ptr<Pruner> _pruner;

    // === Centroid data ===
    // TODO(@lkuffo, high): 3 copies of the centroids? Can we do better?
    //    We can trivially avoid _partial_horizontal_centroids by using the full horizontal
    //    centroids in PDXearch. We can also avoid _prev_centroids if we dont care about the shift
    //    convergence check.
    std::vector<centroid_value_t> _centroids;                    // PDX-layout centroids
    std::vector<centroid_value_t> _horizontal_centroids;         // Row-major centroids
    std::vector<centroid_value_t> _prev_centroids;               // Previous iteration centroids
    std::vector<centroid_value_t> _partial_horizontal_centroids; // First _vertical_d dimensions

    // === Assignment and distance data ===
    std::vector<distance_t> _distances;
    std::vector<uint32_t> _cluster_sizes;
    std::vector<vector_value_t> _data_norms;
    std::vector<vector_value_t> _centroid_norms;

    // === Ground truth and recall computation ===
    std::vector<uint32_t> _gt_assignments;
    std::vector<distance_t> _gt_distances;
    std::vector<distance_t> _query_norms;

    // === Reusable buffers (avoid repeated allocations) ===
    std::vector<distance_t> _tmp_distances_buffer;
    std::vector<uint32_t> _promising_centroids;
    std::vector<distance_t> _recall_distances;

  public:
    std::vector<uint32_t> _assignments; // Public for user access
    std::vector<SuperKMeansIterationStats>
        iteration_stats; // Public for user access to iteration statistics
};
} // namespace skmeans
