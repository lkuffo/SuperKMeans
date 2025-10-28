#pragma once
#ifndef SUPERKMEANS_H
#define SUPERKMEANS_H

#include "superkmeans/common.h"
#include "superkmeans/distance_computers/base_computers.hpp"
#include "superkmeans/pdx/pdxearch.h"
#include "superkmeans/pdx/utils.h"

namespace skmeans {
template <Quantization q = f32, DistanceFunction alpha = l2, class Pruner = ADSamplingPruner<q>>
class SuperKMeans {
    using centroid_value_t = skmeans_centroid_value_t<q>;
    using vector_value_t = skmeans_value_t<q>;
    using layout_t = PDXLayout<q, alpha, Pruner>;
    using knn_candidate_t = KNNCandidate<q>;

  public:
    // TODO(@lkuffo, high): N-threads should be controlled with a global function. By default
    //   it is set to all threads
    SuperKMeans(
        size_t n_clusters, size_t dimensionality, uint32_t iters = 25,
        float sampling_fraction = 0.50
    )
        : _iters(iters), _n_clusters(n_clusters), _sampling_fraction(sampling_fraction),
          _d(dimensionality), _trained(false), verbose(false) {
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
    std::vector<skmeans_centroid_value_t<q>>
    Train(const vector_value_t* SKM_RESTRICT data, const size_t n, uint32_t num_threads = -1) {
        SKMEANS_ENSURE_POSITIVE(n);
        if (_trained) {
            throw std::runtime_error("The clustering has already been trained");
        }

        if (n < _n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters"
            );
        }

        vector_value_t* SKM_RESTRICT data_p = data;
        _centroids.resize(_n_clusters * _d);
        _tmp_centroids.resize(_n_clusters * _d);
        _cluster_sizes.resize(_n_clusters);
        _reciprocal_cluster_sizes.resize(_n_clusters);
        /*
         * 1. [done] Sample using sampling_fraction. In this case you need to create a new buffer
         * 2. [todo] If metric is dp, normalize the vectors so you can use l2 always
         * 3. [done] Transform data in data_p with Pruner preprocessing
         *    - ADSampling: Returns pointer to new buffer     [done]
         *    - Flat      : Zero copy (pass the pointer back) [done]
         * 4. [done] Sample centroids from data, copy them in the centroid buffer
         * 5. [done] Transform data in centroids with Pruner preprocessing
         *    - ADSampling  [done]
         *    - Flat        [done]
         * 6. [done] PDXify centroids! (I don't need to PDXify data)
         * 7. [next] Loop: assign+update, split
         * 8. Finalize:
         *    - I need to unrotate the centroids (Do assignments change if I derotate?)
         *    - I am not sure if I should just return the assignments
         *    - Assigning later would mean that I need to do the rotation also later
         */

        if (verbose) {
            std::cout << "Sampling data..." << std::endl;
        }
        //! _centroids is always wrapped with the PDXLayout object
        auto centroids_pdx_wrapper = GenerateCentroids(data_p, n);
        const auto n_samples = GetNVectorsToSample(n);
        auto data_to_cluster = SampleVectors(data_p, n, n_samples);

        if (verbose) {
            std::cout << "Clustering..." << std::endl;
        }

        for (int i = 0; i < _iters; ++i) {
            AssignAndUpdateCentroids(data_to_cluster, centroids_pdx_wrapper, n_samples);
            SplitClusters();
            if (alpha == dp) {
                PostprocessCentroids();
            }
            if (verbose)
                std::cout << "Iteration " << i + 1 << "/" << _iters
                          << " | Objective: " << std::endl;
        }
        //! I don't need proper assignments until the last iteration
        Assign(data, centroids_pdx_wrapper, n);

        _trained = true;
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
    std::vector<uint32_t> AssignAndUpdateCentroids(
        const vector_value_t* SKM_RESTRICT data, const layout_t pdx_centroids, const size_t n
    ) {
        // Data is a row-major matrix
        // Centroids is already in the PDX layout:
        //      Maybe we should use the PDXLayout abstraction and call Search() per vector
        //      Random access, no IVF... for now!
        // But, since collection is already rotated I need some hacks
        // Anyways...
        // fork::union
        // For each vector in data:
        //      0. What if I start with a 1 to 1 distance with the same centroid that was assigned
        //      last time and I set it as an initial lower bound?
        //           This would be more effective than doing a mini-kmeans on the centroids
        //           But not sure how good
        std::fill(_tmp_centroids.begin(), _tmp_centroids.end(), 0.0);
        std::fill(_cluster_sizes.begin(), _cluster_sizes.end(), 0);
        auto data_p = data;
        // TODO(@lkuffo, high): Remove this out
        auto [horizontal_d, vertical_d] = PDXLayout<q, alpha>::GetDimensionSplit(_d);
        for (size_t i = 0; i < n; ++i) {
            // PDXearch per vector
            std::vector<knn_candidate_t> assignment = pdx_centroids.searcher->Search(data_p, 1);
            auto assignment_idx = assignment[0].index;
            _cluster_sizes[assignment_idx] += 1;
            // TODO(@lkuffo, med): Is it better to update directly on PDX or keep as RowMaj+PDXify
            UpdateCentroid(data_p, assignment_idx);
            data_p += _d;
        }
        ConsolidateCentroids();
    }

    std::vector<uint32_t>
    Assign(const vector_value_t* SKM_RESTRICT data, const layout_t pdx_centroids, const size_t n) {
        // TODO(@lkuffo, low): What if I start with a 1 to 1 distance with the same centroid that
        //      was assigned last time and I set it as an initial lower bound?
        //      I can force the Start() to that chunk?
        auto data_p = data;
        // TODO(@lkuffo, med): Skip the vectors that were used as samples
        for (size_t i = 0; i < n; ++i) {
            // PDXearch per vector
            std::vector<knn_candidate_t> assignment = pdx_centroids.searcher->Search(data_p, 1);
            auto assignment_idx = assignment[0].index;
            data_p += _d;
        }
        // TODO(@lkuffo, medium): Do I need to return the true centroids?
    }

    // TODO(@lkuffo, critical)
    void SplitClusters() {
        for (size_t i = 0; i < _n_clusters; ++i) {
            if (_cluster_sizes[i] == 0) {
                std::cout << "Need to split centroid: " << i << std::endl;
            }
        }
    }

    SKM_ALWAYS_INLINE void
    UpdateCentroid(vector_value_t* SKM_RESTRICT vector, const uint32_t cluster_idx) {
        // Potentially also store the information to quickly calculate the norms for DP
        // TODO(@lkuffo, low): This should be trivially auto-vectorized in any architecture
        for (size_t i = 0; i < _d; ++i) {
            _tmp_centroids[cluster_idx * _d + i] += vector[i];
        }
    }

    void ConsolidateCentroids() {
        for (size_t i = 0; i < _cluster_sizes.size(); ++i) {
            _reciprocal_cluster_sizes[i] = 1.0 / _cluster_sizes[i];
        }
        auto _tmp_centroids_p = _tmp_centroids.data();
        for (size_t i = 0; i < _n_clusters; ++i) {
            auto mult_factor = _reciprocal_cluster_sizes[i];
            // TODO(@lkuffo, low): This should be trivially auto-vectorized in any architecture
            for (size_t j = 0; j < _d; ++j) {
                _tmp_centroids_p[j] = _tmp_centroids_p[j] * mult_factor;
            }
            _tmp_centroids_p += _d;
        }
        // memcpy and PDXify in one go
        PDXLayout<q, alpha, Pruner>::PDXify<q, true>(
            _tmp_centroids.data(), _centroids.data(), _n_clusters, _d
        );
    }

    std::vector<skmeans_centroid_value_t<q>> GetCentroids() const { return _centroids; }

    inline size_t GetNClusters() const { return _n_clusters; }

    inline bool IsTrained() const { return _trained; }

  protected:
    // Equidistant sampling similar to DuckDB's
    PDXLayout<q, alpha, Pruner>
    GenerateCentroids(vector_value_t* SKM_RESTRICT data, const size_t n) const {
        const auto jumps = static_cast<size_t>(std::floor(1.0 * n / _n_clusters));
        auto tmp_centroids_p = _tmp_centroids.data();
        for (size_t i = 0; i < n; i += jumps) {
            // TODO(@lkuffo, low): What if centroid scalar_t are not the same size of vector ones
            memcpy(tmp_centroids_p, data + i, sizeof(centroid_value_t) * _d);
            tmp_centroids_p += _d;
        }
        // We populate the _centroids buffer with the centroids in the PDX layout
        if constexpr (std::is_same_v<Pruner, ADSamplingPruner<q>>) {
            // TODO(@lkuffo, high): Implement a template bool for pruning or not, this would depend
            //    on the dimensionality of the data
            std::vector<centroid_value_t> rotated_centroids(_n_clusters * _d);
            _pruner->Rotate(_tmp_centroids.data(), rotated_centroids.data(), _n_clusters);
            PDXLayout<q, alpha, Pruner>::PDXify<q, true>(
                rotated_centroids.data(), _centroids.data(), _n_clusters, _d
            );
        } else {
            PDXLayout<q, alpha, Pruner>::PDXify<q, true>(
                tmp_centroids_p.data(), _centroids.data(), _n_clusters, _d
            );
        }
        //! We wrap _centroids in the PDXLayout wrapper
        auto pdx_centroids =
            PDXLayout<q, alpha, Pruner>(_centroids.data(), _pruner, _n_clusters, _d);
        return pdx_centroids;
    }

    size_t GetNVectorsToSample(const size_t n) const {
        if (_sampling_fraction == 1.0) { // Needed?
            return n;
        }
        return std::floor(n * _sampling_fraction);
    }

    // TODO(@lkuffo, high): Centroids are on PDX, I cant do this
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
    vector_value_t*
    SampleVectors(vector_value_t* SKM_RESTRICT data, const size_t n, const size_t n_sampled) const {
        vector_value_t* tmp_data_buffer_p;
        // TODO(@lkuffo, medium): If DP, normalize here while taking the samples
        if (n_sampled < n) {
            std::vector<vector_value_t> tmp_data_buffer(n_sampled * _d);
            tmp_data_buffer_p = tmp_data_buffer.data();
            const auto jumps = static_cast<size_t>(std::floor((1.0 * n) / n_sampled));
            auto tmp_data_p = data;
            for (size_t i = 0; i < n_sampled; i += jumps) {
                memcpy(tmp_data_buffer_p, data + i, sizeof(vector_value_t) * _d);
                tmp_data_buffer_p += _d;
            }
        } else {                      // Flat
            tmp_data_buffer_p = data; // Zero-copy
        }

        if constexpr (std::is_same_v<Pruner, ADSamplingPruner<q>>) {
            // TODO(@lkuffo, high): Try to remove temporary buffer for rotating the vectors (sad)
            std::vector<vector_value_t> rotated_data(n_sampled * _d);
            _pruner->Rotate(tmp_data_buffer_p, rotated_data.data(), n_sampled);
            return rotated_data.data();
        } else {                      // Flat
            return tmp_data_buffer_p; // Zero-copy
        }
    }

    Pruner _pruner;

    std::vector<centroid_value_t> _centroids;
    std::vector<centroid_value_t> _tmp_centroids;
    std::vector<centroid_value_t> _super_centroids;
    std::vector<uint32_t> _assignments;
    std::vector<uint32_t> _cluster_sizes;
    std::vector<float> _reciprocal_cluster_sizes;

    const uint32_t _iters;
    const size_t _n_clusters;
    const float _sampling_fraction;
    const size_t _d;
    bool _trained;
    bool verbose;
};
} // namespace skmeans

#endif // SUPERKMEANS_H
