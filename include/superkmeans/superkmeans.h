#pragma once

#include <Eigen/Dense>
#include <iomanip>
#include <omp.h>
#include <random>

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/pdx/pdxearch.h"
#include "superkmeans/pdx/utils.h"
#include "superkmeans/profiler.h"

namespace skmeans {

/**
 * @brief Configuration parameters for SuperKMeans clustering.
 * Can be passed to the SuperKMeans constructor.
 *
 */
struct SuperKMeansConfig {
    // Training parameters
    uint32_t iters = 10; // Number of k-means iterations
    // We provide 2 ways to define the number of points to sample:
    float sampling_fraction = 0.3f; // Fraction of data to sample (0.0 to 1.0)
    uint32_t max_points_per_cluster =
        256;                    // Maximum number of points per cluster to sample (FAISS style)
    uint32_t n_threads = 0;     // Number of CPU threads (0 = max)
    uint32_t seed = 42;         // Random seed for reproducibility
    bool use_blas_only = false; // Use BLAS-only computation for all iterations

    // Convergence parameters
    float tol = 1e-4f;                  // Tolerance for shift-based early termination
    float recall_tol = 0.005f;          // Tolerance for recall-based early termination
    bool early_termination = true;      // Whether to stop early on convergence
    bool sample_queries = false;        // Whether to sample queries from data
    size_t objective_k = 100;           // Number of nearest neighbors for recall computation
    float ann_explore_fraction = 0.01f; // Fraction of centroids to explore (0.0 to 1.0)

    // Sweet range for d' tuning
    float min_not_pruned_pct = 0.03f; // Minimum percentage of vectors not pruned (3% = 97% pruned)
    float max_not_pruned_pct = 0.05f; // Maximum percentage of vectors not pruned (5% = 95% pruned)
    // Adjustment factor for d' tuning when outside the sweet range
    float adjustment_factor_for_partial_d = 0.20f;

    // Output parameters
    bool unrotate_centroids = true;   // Whether to unrotate centroids before returning
    bool perform_assignments = false; // Whether to perform final assignment on Train()
    bool verbose = false;             // Whether to print progress information
    bool angular = false;             // Whether to use spherical k-means

    bool data_already_rotated = false; // Whether input data is already rotated (skip rotation)
};

/**
 * @brief Statistics for a single iteration of SuperKMeans clustering.
 */
struct SuperKMeansIterationStats {
    size_t iteration = 0;   // Iteration number (1-indexed)
    float objective = 0.0f; // Total clustering cost (WCSS)
    float shift = 0.0f;     // Average squared centroid shift from previous iteration
    size_t split = 0;       // Number of clusters that were split (empty cluster handling)
    float recall = 0.0f;    // Recall@k value (0.0 to 1.0, only when queries provided)
    // Percentage of vectors not pruned (0.0 to 1.0, -1.0 if not applicable)
    float not_pruned_pct = -1.0f;
    // Number of dimensions used for partial distance computation (d')
    uint32_t partial_d = 0;
    // Whether this iteration used BLAS-only computation (no PDX pruning)
    bool is_gemm_only = false;
};

/**
 * @brief Statistics about cluster size balance.
 */
struct ClusterBalanceStats {
    float mean = 0.0f;
    float geometric_mean = 0.0f;
    float stdev = 0.0f;
    float cv = 0.0f;
    size_t min = 0;
    size_t max = 0;

    std::string to_json() const {
        std::ostringstream oss;
        oss << "{\"mean\":" << mean << ",\"geometric_mean\":" << geometric_mean
            << ",\"stdev\":" << stdev << ",\"cv\":" << cv << ",\"min\":" << min
            << ",\"max\":" << max << "}";
        return oss.str();
    }

    void print() const {
        std::cout << "Cluster size stats: "
                  << "mean=" << mean << ", gmean=" << geometric_mean << ", std=" << stdev
                  << ", CV=" << cv << ", min=" << min << ", max=" << max << std::endl;
    }
};

template <Quantization q = Quantization::f32, DistanceFunction alpha = DistanceFunction::l2>
class SuperKMeans {
  protected:
    using centroid_value_t = skmeans_centroid_value_t<q>;
    using vector_value_t = skmeans_value_t<q>;
    using pruner_t = ADSamplingPruner<q>;
    using layout_t = PDXLayout<q, alpha>;
    using distance_t = skmeans_distance_t<q>;
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using VectorR = Eigen::VectorXf;
    using batch_computer = BatchComputer<alpha, q>;

  public:
    /**
     * @brief Constructor with custom configuration
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
        _n_threads = (_config.n_threads == 0) ? omp_get_max_threads() : _config.n_threads;
        g_n_threads = _n_threads;
        _pruner =
            std::make_unique<pruner_t>(dimensionality, PRUNER_INITIAL_THRESHOLD, _config.seed);

        // If data is already rotated, we must not unrotate output centroids
        if (_config.data_already_rotated) {
            _config.unrotate_centroids = false;
        }
    }

    /**
     * @brief Default constructor
     *
     * @param n_clusters Number of clusters to create
     * @param dimensionality Number of dimensions in the data
     */
    SuperKMeans(size_t n_clusters, size_t dimensionality)
        : SuperKMeans(n_clusters, dimensionality, SuperKMeansConfig{}) {}

    /**
     * @brief Run k-means clustering to determine centroids
     *
     * @param data Pointer to the data matrix (row-major, n × d)
     * @param n Number of points (rows) in the data matrix
     * @param queries Optional pointer to query vectors for recall computation
     * @param n_queries Number of query vectors (ignored if queries is nullptr and sample_queries is
     * false)
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
        _n_samples = GetNVectorsToSample(n, _n_clusters);
        if (_n_samples < _n_clusters) {
            throw std::runtime_error(
                "Not enough samples to train. Try increasing the sampling_fraction or "
                "max_points_per_cluster"
            );
        }
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
        std::vector<vector_value_t> centroids_partial_norms(_n_clusters);
        std::vector<size_t> not_pruned_counts(_n_samples);
        std::vector<distance_t> tmp_distances_buf(X_BATCH_SIZE * Y_BATCH_SIZE);
        _vertical_d = PDXLayout<q, alpha>::GetDimensionSplit(_d).vertical_d;
        _partial_horizontal_centroids.resize(_n_clusters * _vertical_d);

        // Set _partial_d (d') dynamically as half of _vertical_d (around 12% of d)
        _partial_d = std::max<uint32_t>(MIN_PARTIAL_D, _vertical_d / 2);
        if (_partial_d > _vertical_d) {
            _partial_d = _vertical_d;
        }
        if (_config.verbose) {
            std::cout << "Front dimensions (d') = " << _partial_d << std::endl;
            std::cout << "Trailing dimensions (d'') = " << _d - _vertical_d << std::endl;
        }

        auto centroids_pdx_wrapper =
            GenerateCentroids(data_p, _n_samples, _n_clusters, !_config.data_already_rotated);
        if (_config.verbose) {
            std::cout << "Sampling data..." << std::endl;
        }

        std::vector<vector_value_t> data_samples_buffer;
        SampleAndRotateVectors(
            data_p, data_samples_buffer, n, _n_samples, !_config.data_already_rotated
        );
        auto data_to_cluster = data_samples_buffer.data();

        RotateOrCopy(
            _horizontal_centroids.data(),
            _prev_centroids.data(),
            _n_clusters,
            !_config.data_already_rotated
        );

        GetL2NormsRowMajor(data_to_cluster, _n_samples, _data_norms.data());
        GetL2NormsRowMajor(_prev_centroids.data(), _n_clusters, _centroid_norms.data());

        std::vector<vector_value_t> rotated_queries;
        if (n_queries) {
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
            rotated_queries.resize(n_queries * _d);
            if (_config.sample_queries) {
                std::cout << "Sampling queries from data..." << std::endl;
                SampleAndRotateVectors(
                    data_to_cluster, rotated_queries, _n_samples, n_queries, false
                );
            } else {
                RotateOrCopy(
                    queries, rotated_queries.data(), n_queries, !_config.data_already_rotated
                );
            }
            _query_norms.resize(n_queries);
            GetL2NormsRowMajor(rotated_queries.data(), n_queries, _query_norms.data());
            GetGTAssignmentsAndDistances(data_to_cluster, rotated_queries.data(), n_queries);
        }

        //
        // 1st iteration: FULL GEMM
        //
        if (_config.verbose)
            std::cout << "1st iteration..." << std::endl;
        size_t iter_idx = 0;
        float best_recall = 0.0f;
        size_t iters_without_improvement = 0;
        RunIteration<true>(
            data_to_cluster,
            tmp_distances_buf.data(),
            centroids_pdx_wrapper,
            centroids_partial_norms,
            not_pruned_counts,
            rotated_queries.data(),
            n_queries,
            _n_samples,
            _n_clusters,
            iter_idx,
            true, // is_first_iter
            iteration_stats
        );
        iter_idx = 1;
        best_recall = _recall;
        if (_config.iters <= 1) {
            auto output_centroids = GetOutputCentroids(_config.unrotate_centroids);
            if (_config.perform_assignments) {
                _assignments = Assign(data, output_centroids.data(), n, _n_clusters);
            }
            if (_config.verbose) {
                Profiler::Get().PrintHierarchical();
            }
            return output_centroids;
        }

        //
        // FULL GEMM on low-dimensional data or too few clusters
        //
        if (_d < DIMENSION_THRESHOLD_FOR_PRUNING || _config.use_blas_only ||
            _n_clusters <= N_CLUSTERS_THRESHOLD_FOR_PRUNING) {
            for (; iter_idx < _config.iters; ++iter_idx) {
                RunIteration<true>(
                    data_to_cluster,
                    tmp_distances_buf.data(),
                    centroids_pdx_wrapper,
                    centroids_partial_norms,
                    not_pruned_counts,
                    rotated_queries.data(),
                    n_queries,
                    _n_samples,
                    _n_clusters,
                    iter_idx,
                    false, // is_first_iter
                    iteration_stats
                );
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

        //
        // Rest of iterations with GEMM+PRUNING
        //
        GetPartialL2NormsRowMajor(data_to_cluster, _n_samples, _data_norms.data(), _partial_d);
        for (; iter_idx < _config.iters; ++iter_idx) {
            RunIteration<false>(
                data_to_cluster,
                tmp_distances_buf.data(),
                centroids_pdx_wrapper,
                centroids_partial_norms,
                not_pruned_counts,
                rotated_queries.data(),
                n_queries,
                _n_samples,
                _n_clusters,
                iter_idx,
                false, // is_first_iter
                iteration_stats
            );
            if (_config.early_termination &&
                ShouldStopEarly(n_queries > 0, best_recall, iters_without_improvement, iter_idx)) {
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

    /**
     * @brief Assign vectors to their nearest centroid.
     *
     * Both vectors and centroids are assumed to be in the same domain
     * (no rotation/transformation needed).
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

        std::vector<vector_value_t> vector_norms(n_vectors);
        std::vector<vector_value_t> centroid_norms_local(n_centroids);
        std::vector<uint32_t> assignments(n_vectors);
        std::vector<distance_t> distances(n_vectors);
        std::vector<distance_t> tmp_distances_buf(X_BATCH_SIZE * Y_BATCH_SIZE);

        Eigen::Map<const MatrixR> vectors_mat(vectors, n_vectors, _d);
        Eigen::Map<VectorR> v_norms(vector_norms.data(), n_vectors);
        v_norms.noalias() = vectors_mat.rowwise().squaredNorm();

        Eigen::Map<const MatrixR> centroids_mat(centroids, n_centroids, _d);
        Eigen::Map<VectorR> c_norms(centroid_norms_local.data(), n_centroids);
        c_norms.noalias() = centroids_mat.rowwise().squaredNorm();

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
            tmp_distances_buf.data()
        );

        return assignments;
    }

    /** @brief Returns the number of clusters. */
    [[nodiscard]] inline size_t GetNClusters() const noexcept { return _n_clusters; }

    /** @brief Returns whether the model has been trained. */
    [[nodiscard]] inline bool IsTrained() const noexcept { return _trained; }

    /**
     * @brief Calculate cluster balance statistics from assignments
     *
     * @param assignments Array of cluster assignments [n_samples]
     * @param n_samples Number of samples
     * @param n_clusters Number of clusters
     * @return ClusterBalanceStats containing mean, stdev, CV, min, max
     */
    [[nodiscard]] static ClusterBalanceStats GetClustersBalanceStats(
        const uint32_t* assignments,
        size_t n_samples,
        size_t n_clusters
    ) {
        ClusterBalanceStats stats;
        std::vector<size_t> cluster_sizes(n_clusters, 0);
        for (size_t i = 0; i < n_samples; ++i) {
            cluster_sizes[assignments[i]]++;
        }

        float sum = std::accumulate(cluster_sizes.begin(), cluster_sizes.end(), 0.0f);
        stats.mean = sum / cluster_sizes.size();

        // Geometric mean
        float log_sum = 0.0f;
        size_t non_zero_count = 0;
        for (size_t size : cluster_sizes) {
            if (size > 0) {
                log_sum += std::log(static_cast<float>(size));
                non_zero_count++;
            }
        }
        stats.geometric_mean = (non_zero_count > 0) ? std::exp(log_sum / non_zero_count) : 0.0f;

        float sq_sum = std::inner_product(
            cluster_sizes.begin(), cluster_sizes.end(), cluster_sizes.begin(), 0.0f
        );
        stats.stdev = std::sqrt(sq_sum / cluster_sizes.size() - stats.mean * stats.mean);

        // Coefficient of variation
        stats.cv = stats.stdev / stats.mean;

        auto minmax = std::minmax_element(cluster_sizes.begin(), cluster_sizes.end());
        stats.min = *minmax.first;
        stats.max = *minmax.second;

        return stats;
    }

  protected:
    /**
     * @brief Performs first assignment and centroid update using FULL GEMM.
     *
     * Used for the first iteration where full distance computation via GEMM is used
     * (no pruning). Assigns each data point to its nearest centroid, then updates
     * centroid positions.
     *
     * @param data Data matrix (row-major, _n_samples × _d)
     * @param rotated_initial_centroids Initial centroids (row-major, _n_clusters × _d)
     * @param tmp_distances_buf Workspace buffer for distance computations
     * @param n_samples Number of vectors in the data
     * @param n_clusters Number of centroids
     */
    void FirstAssignAndUpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT rotated_initial_centroids,
        distance_t* SKM_RESTRICT tmp_distances_buf,
        const size_t n_samples,
        const size_t n_clusters
    ) {
        batch_computer::FindNearestNeighbor(
            data,
            rotated_initial_centroids,
            n_samples,
            n_clusters,
            _d,
            _data_norms.data(),
            _centroid_norms.data(),
            _assignments.data(),
            _distances.data(),
            tmp_distances_buf
        );
        {
            SKM_PROFILE_SCOPE("fill");
            std::fill(
                _horizontal_centroids.data(),
                _horizontal_centroids.data() + (n_clusters * _d),
                0.0
            );
            std::fill(_cluster_sizes.data(), _cluster_sizes.data() + n_clusters, 0);
        }
    }

    /**
     * @brief Performs assignment and centroid update using GEMM+PRUNING.
     *
     * Uses GEMM for partial distance computation (first _partial_d dimensions),
     * then PRUNING for completing distances for remaining candidates.
     *
     * @param data Data matrix (row-major, _n_samples × _d)
     * @param centroids Centroids to use for GEMM distance computation (row-major)
     * @param partial_centroid_norms Partial norms of centroids (first _partial_d dims)
     * @param tmp_distances_buf Workspace buffer for distance computations
     * @param pdx_centroids PDX-layout centroids for PRUNING
     * @param out_not_pruned_counts Output for pruning statistics
     */
    void AssignAndUpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const vector_value_t* SKM_RESTRICT centroids,
        const vector_value_t* SKM_RESTRICT partial_centroid_norms,
        distance_t* SKM_RESTRICT tmp_distances_buf,
        const layout_t& pdx_centroids,
        size_t* out_not_pruned_counts,
        const size_t n_samples,
        const size_t n_clusters
    ) {
        batch_computer::FindNearestNeighborWithPruning(
            data,
            centroids,
            n_samples,
            n_clusters,
            _d,
            _data_norms.data(),
            partial_centroid_norms,
            _assignments.data(),
            _distances.data(),
            tmp_distances_buf,
            pdx_centroids,
            _partial_d,
            out_not_pruned_counts
        );
        {
            SKM_PROFILE_SCOPE("fill");
            std::fill(
                _horizontal_centroids.data(),
                _horizontal_centroids.data() + (n_clusters * _d),
                0.0
            );
            std::fill(_cluster_sizes.data(), _cluster_sizes.data() + n_clusters, 0);
        }
    }

    /**
     * @brief Updates centroids by accumulating assigned vectors.
     *
     * After this call, _horizontal_centroids contains the sum of assigned vectors.
     * ConsolidateCentroids() must be called to normalize by cluster sizes.
     *
     * @param data Data matrix (row-major, _n_samples × _d)
     * @param n_samples
     * @param n_clusters
     */
    void UpdateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n_samples,
        const size_t n_clusters
    ) {
        SKM_PROFILE_SCOPE("update_centroids");
#pragma omp parallel if (_n_threads > 1) num_threads(_n_threads)
        {
            uint32_t nt = _n_threads;
            uint32_t rank = omp_get_thread_num();
            // This thread is taking care of centroids c0:c1
            size_t c0 = (n_clusters * rank) / nt;
            size_t c1 = (n_clusters * (rank + 1)) / nt;
            for (size_t i = 0; i < n_samples; i++) {
                uint32_t ci = _assignments[i];
                assert(ci < n_clusters);
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
     * @brief Runs a single K-Means iteration with either GEMM-only or GEMM+PRUNING strategy.
     *
     *
     * @tparam GEMM_ONLY If true, uses full GEMM (FirstAssignAndUpdateCentroids).
     *                   If false, uses GEMM+PRUNING (AssignAndUpdateCentroids with TunePartialD).
     *
     * @param data_to_cluster Training data (rotated, row-major)
     * @param tmp_distances_buf Workspace buffer for distance computations
     * @param centroids_pdx_wrapper PDX-layout centroids (only used when !GEMM_ONLY)
     * @param centroids_partial_norms Partial norms buffer (only used when !GEMM_ONLY)
     * @param not_pruned_counts Pruning statistics buffer (only used when !GEMM_ONLY)
     * @param rotated_queries Query vectors for recall computation (nullptr if n_queries==0)
     * @param n_queries Number of query vectors
     * @param n_samples Number of training samples
     * @param n_clusters Number of clusters
     * @param iter_idx Current iteration index (0-based)
     * @param is_first_iter Whether this is the first iteration (skips centroid swap)
     */
    template <bool GEMM_ONLY>
    void RunIteration(
        const vector_value_t* SKM_RESTRICT data_to_cluster,
        distance_t* SKM_RESTRICT tmp_distances_buf,
        const layout_t& centroids_pdx_wrapper,
        std::vector<vector_value_t>& centroids_partial_norms,
        std::vector<size_t>& not_pruned_counts,
        const vector_value_t* SKM_RESTRICT rotated_queries,
        const size_t n_queries,
        const size_t n_samples,
        const size_t n_clusters,
        size_t& iter_idx,
        const bool is_first_iter,
        std::vector<SuperKMeansIterationStats>& target_stats
    ) {
        if (!is_first_iter) {
            std::swap(_horizontal_centroids, _prev_centroids);
        }

        if constexpr (GEMM_ONLY) {
            GetL2NormsRowMajor(_prev_centroids.data(), n_clusters, _centroid_norms.data());
        } else {
            GetPartialL2NormsRowMajor(
                _prev_centroids.data(), n_clusters, centroids_partial_norms.data(), _partial_d
            );
        }

        if constexpr (GEMM_ONLY) {
            FirstAssignAndUpdateCentroids(
                data_to_cluster, _prev_centroids.data(), tmp_distances_buf, n_samples, n_clusters
            );
        } else {
            {
                SKM_PROFILE_SCOPE("fill");
                std::fill(not_pruned_counts.data(), not_pruned_counts.data() + n_samples, 0);
            }
            AssignAndUpdateCentroids(
                data_to_cluster,
                _prev_centroids.data(),
                centroids_partial_norms.data(),
                tmp_distances_buf,
                centroids_pdx_wrapper,
                not_pruned_counts.data(),
                n_samples,
                n_clusters
            );
        }

        UpdateCentroids(data_to_cluster, n_samples, n_clusters);

        float avg_not_pruned_pct = -1.0f;
        uint32_t old_partial_d = _partial_d;
        if constexpr (!GEMM_ONLY) {
            bool partial_d_changed = false;
            avg_not_pruned_pct =
                TunePartialD(not_pruned_counts.data(), n_samples, n_clusters, partial_d_changed);
            if (partial_d_changed) {
                GetPartialL2NormsRowMajor(
                    data_to_cluster, n_samples, _data_norms.data(), _partial_d
                );
            }
        }

        ConsolidateCentroids(n_samples, n_clusters);

        ComputeCost(n_samples);
        ComputeShift(n_clusters);

        if (n_queries) {
            GetL2NormsRowMajor(_horizontal_centroids.data(), n_clusters, _centroid_norms.data());
            _recall = ComputeRecall(rotated_queries, n_queries);
        }

        SuperKMeansIterationStats stats;
        stats.iteration = iter_idx + 1;
        stats.objective = _cost;
        stats.shift = _shift;
        stats.split = _n_split;
        stats.recall = _recall;
        stats.is_gemm_only = GEMM_ONLY;
        if constexpr (!GEMM_ONLY) {
            stats.not_pruned_pct = avg_not_pruned_pct;
            stats.partial_d = old_partial_d;
        }
        target_stats.push_back(stats);

        if (_config.verbose) {
            std::cout << "Iteration " << iter_idx + 1 << "/" << _config.iters
                      << " | Objective: " << _cost << " | Objective improvement: "
                      << (iter_idx > 0 ? 1 - (_cost / _prev_cost) : 0.0f) << " | Shift: " << _shift
                      << " | Split: " << _n_split << " | Recall: " << _recall;
            if constexpr (GEMM_ONLY) {
                std::cout << " [BLAS-only]";
            } else {
                std::cout << " | Not Pruned %: " << avg_not_pruned_pct * 100.0f
                          << " | d': " << old_partial_d << " -> " << _partial_d;
            }
            std::cout << std::endl << std::endl;
        }
    }

    /**
     * @brief Handles empty clusters by splitting large clusters.
     * Taken from FAISS implementation:
     * https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.cpp
     *
     * When a cluster becomes empty (no points assigned), this method splits
     * a large cluster to repopulate it. Selection is probabilistic based on
     * cluster sizes.
     */
    virtual void SplitClusters(const size_t n_samples, const size_t n_clusters) {
        _n_split = 0;
        std::mt19937 rng(_config.seed);
        auto _horizontal_centroids_p = _horizontal_centroids.data();
        for (size_t ci = 0; ci < n_clusters; ci++) {
            if (_cluster_sizes[ci] == 0) {
                size_t cj;
                for (cj = 0; true; cj = (cj + 1) % n_clusters) {
                    // Probability to pick this cluster for split
                    float p = (_cluster_sizes[cj] - 1.0) / (float) (n_samples - n_clusters);
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
    void ConsolidateCentroids(const size_t n_samples, const size_t n_clusters) {
        SKM_PROFILE_SCOPE("consolidate");
        {
            SKM_PROFILE_SCOPE("consolidate/splitting");
#pragma omp parallel for if (_n_threads > 1) num_threads(_n_threads)
            for (size_t i = 0; i < n_clusters; ++i) {
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
            SplitClusters(n_samples, n_clusters);
        }
        {
            SKM_PROFILE_SCOPE("consolidate/normalize");
            if (_config.angular) {
                PostprocessCentroids(n_clusters);
            }
        }
        {
            SKM_PROFILE_SCOPE("consolidate/pdxify");
            //! This updates the object within the pdx_layout wrapper
            PDXLayout<q, alpha>::template PDXify<false>(
                _horizontal_centroids.data(), _centroids.data(), n_clusters, _d
            );
            CentroidsToAuxiliaryHorizontal(n_clusters);
        }
    }

    /**
     * @brief Computes Within-Cluster Sum of Squares (WCSS).
     */
    void ComputeCost(const size_t n_samples) {
        SKM_PROFILE_SCOPE("compute_cost");
        _prev_cost = _cost;
        _cost = 0.0f;
#pragma clang loop vectorize(enable)
        for (size_t i = 0; i < n_samples; ++i) {
            _cost += _distances[i];
        }
    }

    /**
     * @brief Computes the squared centroid shift from previous iteration.
     *
     * Used for convergence detection - small shift indicates centroids have stabilized.
     */
    void ComputeShift(const size_t n_clusters) {
        SKM_PROFILE_SCOPE("shift");
        Eigen::Map<const MatrixR> new_mat(_horizontal_centroids.data(), n_clusters, _d);
        Eigen::Map<const MatrixR> prev_mat(_prev_centroids.data(), n_clusters, _d);
        float shift = 0.0f;
#pragma omp parallel for reduction(+ : shift) if (_n_threads > 1) num_threads(_n_threads)
        for (size_t i = 0; i < n_clusters; ++i) {
            shift += (new_mat.row(i) - prev_mat.row(i)).squaredNorm();
        }
        _shift = shift;
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
        batch_computer::FindKNearestNeighbors(
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
        batch_computer::FindKNearestNeighbors(
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
        // top-%centroids_to_explore assignments
        // Recall per query = (# matched GT assignments in top-%centroids_to_explore) / objective_k
        // Final recall = average over all queries
        float sum_recall = 0.0f;
        for (size_t i = 0; i < n_queries; ++i) {
            size_t found_in_query = 0;
            // For each GT assignment for query q
            for (size_t j = 0; j < _config.objective_k; ++j) {
                uint32_t gt = _gt_assignments[i * _config.objective_k + j]; // gt is a vector index
                // Check if this GT assignment is present in the top-%centroids_to_explore
                // assignments for this query
                bool found = false;
                for (size_t t = 0; t < _centroids_to_explore; ++t) {
                    // If a centroid is the same as the GT centroid assignment, then we have a match
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
     * Forgy sampling: Randomly samples _n_clusters vectors as initial centroids,
     * rotates them, PDXifies them, and wraps them in a PDXLayout wrapper.
     *
     * @param data Data matrix
     * @param n_clusters Number of centroids to generate
     * @param rotate Wheter to rotate the sampled centroids
     * @return PDXLayout wrapper for the centroids
     */
    PDXLayout<q, alpha> GenerateCentroids(
        const vector_value_t* SKM_RESTRICT data,
        const size_t n_points,
        const size_t n_clusters,
        const bool rotate = true
    ) {
        {
            SKM_PROFILE_SCOPE("generating_centroids");
            auto tmp_centroids_p = _horizontal_centroids.data();

            std::mt19937 rng(_config.seed);
            std::vector<size_t> indices(n_points);
            for (size_t i = 0; i < n_points; ++i) {
                indices[i] = i;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            for (size_t i = 0; i < n_clusters; i += 1) {
                memcpy(
                    (void*) tmp_centroids_p,
                    (void*) (data + (indices[i] * _d)),
                    sizeof(centroid_value_t) * _d
                );
                tmp_centroids_p += _d;
            }
        }
        // We populate the _centroids buffer with the centroids in the PDX layout
        std::vector<centroid_value_t> rotated_centroids(n_clusters * _d);
        RotateOrCopy(_horizontal_centroids.data(), rotated_centroids.data(), n_clusters, rotate);
        {
            SKM_PROFILE_SCOPE("consolidate/pdxify");
            PDXLayout<q, alpha>::template PDXify<false>(
                rotated_centroids.data(), _centroids.data(), n_clusters, _d
            );
        }
        //! We wrap _centroids and _partial_horizontal_centroids in the PDXLayout wrapper
        //! Any updates to these objects is reflected in the PDXLayout
        //! _partial_horizontal_centroids are not filled until ConsolidateCentroids is called()
        // after the first iteration
        auto pdx_centroids = PDXLayout<q, alpha>(
            _centroids.data(), *_pruner, n_clusters, _d, _partial_horizontal_centroids.data()
        );
        return pdx_centroids;
    }

    /**
     * @brief Computes partial L2 squared norms (first _partial_d dimensions).
     */
    void GetPartialL2NormsRowMajor(
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
     * @brief Rotates or copies vectors based on rotate parameter.
     *
     * If rotate is false, performs a simple memcpy. Otherwise, applies rotation.
     *
     * @param in Input buffer (potentially unrotated)
     * @param out Output buffer (rotated or copied)
     * @param n_vectors Number of vectors to process
     * @param rotate Whether to rotate (true) or just copy (false)
     */
    void RotateOrCopy(
        const centroid_value_t* SKM_RESTRICT in,
        centroid_value_t* SKM_RESTRICT out,
        const size_t n_vectors,
        const bool rotate
    ) {
        SKM_PROFILE_SCOPE("rotator");
        if (rotate) {
            _pruner->Rotate(in, out, n_vectors);
        } else {
            memcpy(
                static_cast<void*>(out),
                static_cast<const void*>(in),
                sizeof(centroid_value_t) * n_vectors * _d
            );
        }
    }

    /**
     * @brief Copies the first _vertical_d dimensions of centroids for efficient PRUNING
     * TODO(@lkuffo, high): We can avoid this by using the full horizontal
     * centroids in PRUNING
     */
    void CentroidsToAuxiliaryHorizontal(const size_t n_clusters) {
        Eigen::Map<MatrixR> hor_centroids(_horizontal_centroids.data(), n_clusters, _d);
        Eigen::Map<MatrixR> out_aux_centroids(
            _partial_horizontal_centroids.data(), n_clusters, _vertical_d
        );
        out_aux_centroids.noalias() = hor_centroids.leftCols(_vertical_d);
    }

    /**
     * @brief Tune _partial_d based on the average not-pruned percentage.
     *
     * A safe range for pruning is between 95% - 97% of vectors pruned (i.e., 3% - 5% not pruned).
     * - If avg_not_pruned_pct > 5% (i.e., less than 95% pruned), we reduce _partial_d by
     * _config.adjustment_factor_for_partial_d to be more aggressive in pruning
     * - If avg_not_pruned_pct < 3% (i.e., more than 97% pruned), we increase _partial_d by
     * _config.adjustment_factor_for_partial_d to be less aggressive
     * - _partial_d is clamped between MIN_PARTIAL_D and _vertical_d
     *
     * @param not_pruned_counts Buffer containing per-vector not-pruned counts
     * @param n_samples Number of X vectors
     * @param n_y Number of Y vectors (centroids)
     * @param partial_d_changed Output parameter: set to true if _partial_d was changed
     * @return The computed average not-pruned percentage
     */
    float TunePartialD(
        const size_t* not_pruned_counts,
        size_t n_samples,
        size_t n_y,
        bool& partial_d_changed
    ) {
        float avg_not_pruned_pct = 0.0f;
        for (size_t i = 0; i < n_samples; ++i) {
            avg_not_pruned_pct += static_cast<float>(not_pruned_counts[i]);
        }
        avg_not_pruned_pct /= static_cast<float>(n_samples * n_y);

        uint32_t old_partial_d = _partial_d;
        if (avg_not_pruned_pct > _config.max_not_pruned_pct) {
            // Too many vectors not pruned (< max_not_pruned_pct pruned), need more GEMM dimensions
            // Increase _partial_d by adjustment_factor_for_partial_d * 2
            auto increase =
                static_cast<uint32_t>(_partial_d * _config.adjustment_factor_for_partial_d * 2);
            _partial_d = std::min(_partial_d + std::max(increase, 1u), _vertical_d);
        } else if (avg_not_pruned_pct < _config.min_not_pruned_pct) {
            // Too few vectors not pruned (> min_not_pruned_pct pruned), can reduce GEMM dimensions
            // Decrease _partial_d by adjustment_factor_for_partial_d
            auto decrease =
                static_cast<uint32_t>(_partial_d * _config.adjustment_factor_for_partial_d);
            _partial_d = std::max(_partial_d - std::max(decrease, 1u), MIN_PARTIAL_D);
        }
        partial_d_changed = (old_partial_d != _partial_d);
        return avg_not_pruned_pct;
    }

    /**
     * @brief Computes the number of vectors to sample based on sampling_fraction.
     * To be conservative, we implement two heuristics:
     * - We sample at most max_points_per_cluster points per cluster (FAISS style)
     * - We sample at most sampling_fraction * n points (our style)
     * We return the minimum of the two.
     * @param n Total number of vectors
     * @return Number of vectors to sample
     */
    [[nodiscard]] virtual size_t GetNVectorsToSample(const size_t n, size_t n_clusters) const {
        if (_config.sampling_fraction == 1.0) {
            return n;
        }
        auto samples_by_n_clusters = n_clusters * _config.max_points_per_cluster;
        auto samples_by_n = static_cast<size_t>(std::floor(n * _config.sampling_fraction));
        return std::min(samples_by_n, samples_by_n_clusters);
    }

    /**
     * @brief Check if the core loop should stop early based on convergence criteria.
     *
     * Convergence is detected when either:
     * - Shift is below tolerance (_shift < _config.tol) or
     * - Recall hasn't improved by more than _config.recall_tol in RECALL_CONVERGENCE_PATIENCE
     * consecutive iterations (when tracking recall)
     *
     * @param tracking_recall Whether recall is being tracked (n_queries > 0)
     * @param best_recall Reference to the best recall seen so far (updated if current is better)
     * @param iters_without_improvement Reference to counter of iterations without recall
     * improvement
     * @param iter_idx Current iteration index
     * @return true if training should stop, false otherwise
     */
    bool ShouldStopEarly(
        const bool tracking_recall,
        float& best_recall,
        size_t& iters_without_improvement,
        const size_t iter_idx
    ) {
        if (_shift < _config.tol) {
            if (_config.verbose)
                std::cout << "Converged at iteration " << iter_idx + 1 << " (shift " << _shift
                          << " < tol " << _config.tol << ")" << std::endl;
            return true;
        }
        if (iter_idx > 0) {
            auto cost_delta = _cost / _prev_cost;
            if (cost_delta > 1 - _config.tol) {
                if (_config.verbose)
                    std::cout << "Converged at iteration " << iter_idx + 1 << " (cost "
                              << " improved by only " << 1 - cost_delta << ")" << std::endl;
                return true;
            }
        }
        if (tracking_recall) {
            float improvement = _recall - best_recall;
            if (improvement > _config.recall_tol) {
                best_recall = _recall;
                iters_without_improvement = 0;
            } else {
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
     * @brief Unrotate centroids to original space.
     * @param should_unrotate If true, unrotates centroids to original space, if false, returns
     * rotated centroids.
     * @return Centroids
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
     * @brief Normalizes centroids to unit length for inner product distance.
     *
     */
    void PostprocessCentroids(const size_t n_clusters) {
        auto horizontal_centroids_p = _horizontal_centroids.data();
#pragma omp parallel for if (_n_threads > 1) num_threads(_n_threads)
        for (size_t i = 0; i < n_clusters; ++i) {
            auto horizontal_centroids_p_i = horizontal_centroids_p + i * _d;
            float sum = 0.0f;
            for (size_t j = 0; j < _d; ++j) {
                sum += horizontal_centroids_p_i[j] * horizontal_centroids_p_i[j];
            }
            float norm = 1.0f / std::sqrt(sum);
            for (size_t j = 0; j < _d; ++j) {
                horizontal_centroids_p_i[j] *= norm;
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
    void SampleAndRotateVectors(
        const vector_value_t* SKM_RESTRICT data,
        std::vector<vector_value_t>& out_buffer,
        const size_t n,
        const size_t n_samples,
        const bool rotate = true
    ) {
        out_buffer.resize(n_samples * _d);

        // Intermediate buffer needed only when both sampling and rotating
        // (we have not yet implemented rotation in-place)
        std::vector<vector_value_t> samples_tmp;
        const vector_value_t* src_data = data;

        if (n_samples < n) {
            if (_config.verbose)
                std::cout << "Sampling " << n_samples << " vectors" << std::endl;
            SKM_PROFILE_SCOPE("sampling");
            // Random sampling without replacement using shuffle
            std::mt19937 rng(_config.seed);
            std::vector<size_t> indices(n);
            {
                SKM_PROFILE_SCOPE("sampling/indices");
                for (size_t i = 0; i < n; ++i) {
                    indices[i] = i;
                }
            }
            {
                SKM_PROFILE_SCOPE("sampling/shuffle");
                std::shuffle(indices.begin(), indices.end(), rng);
            }

            if (rotate) {
                SKM_PROFILE_SCOPE("sampling/memcpy");
                // Need intermediate buffer: sample first, then rotate
                samples_tmp.resize(n_samples * _d);
#pragma omp parallel for if (_n_threads > 1) num_threads(_n_threads)
                for (size_t i = 0; i < n_samples; ++i) {
                    memcpy(
                        static_cast<void*>(samples_tmp.data() + i * _d),
                        static_cast<const void*>(data + indices[i] * _d),
                        sizeof(vector_value_t) * _d
                    );
                }
                src_data = samples_tmp.data();
            } else {
                // No rotation: copy directly into output buffer
#pragma omp parallel for if (_n_threads > 1) num_threads(_n_threads)
                for (size_t i = 0; i < n_samples; ++i) {
                    memcpy(
                        static_cast<void*>(out_buffer.data() + i * _d),
                        static_cast<const void*>(data + indices[i] * _d),
                        sizeof(vector_value_t) * _d
                    );
                }
                return;
            }
        }
        if (_config.verbose)
            std::cout << "Using " << n_samples << " vectors" << std::endl;

        RotateOrCopy(src_data, out_buffer.data(), n_samples, rotate);
    }

    const size_t _d;
    const size_t _n_clusters;
    SuperKMeansConfig _config;

    uint32_t _n_threads;
    size_t _n_samples = 0;
    uint32_t _partial_d = 0; // d'

    // Iteration state
    bool _trained = false;
    size_t _n_split = 0;
    size_t _centroids_to_explore = 0;
    uint32_t _vertical_d = 0;
    float _prev_cost = 0.0f;
    float _cost = 0.0f;
    float _shift = 0.0f;
    float _recall = 0.0f;

    std::unique_ptr<pruner_t> _pruner;

    // Centroids data (unoptimized space)
    // TODO(@lkuffo, high): 3 copies of the centroids? Can we do better?
    //    We can trivially avoid _partial_horizontal_centroids by using the full horizontal
    //    centroids in PDXearch. We can also avoid _prev_centroids if we dont care about the shift
    //    convergence check.
    std::vector<centroid_value_t> _centroids;                    // PDX-layout centroids
    std::vector<centroid_value_t> _horizontal_centroids;         // Row-major centroids
    std::vector<centroid_value_t> _prev_centroids;               // Previous iteration centroids
    std::vector<centroid_value_t> _partial_horizontal_centroids; // First _vertical_d dimensions

    // Buffers for assignment and distance computation
    std::vector<distance_t> _distances;
    std::vector<uint32_t> _cluster_sizes;
    std::vector<vector_value_t> _data_norms;
    std::vector<vector_value_t> _centroid_norms;

    // Buffers for ground truth and recall computation
    std::vector<uint32_t> _gt_assignments;
    std::vector<distance_t> _gt_distances;
    std::vector<distance_t> _query_norms;
    std::vector<distance_t> _tmp_distances_buffer;
    std::vector<uint32_t> _promising_centroids;
    std::vector<distance_t> _recall_distances;

  public:
    std::vector<uint32_t> _assignments;
    std::vector<SuperKMeansIterationStats> iteration_stats;
};
} // namespace skmeans
