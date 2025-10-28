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

  public:
    // @TODO(lkuffo, high): N-threads should be controlled with a global function. By default
    // it is set to all threads
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
        // TODO(@lkuffo): Support non-multiples of 64
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
        SampleCentroids(data_p, n);
        const auto n_samples = GetNVectorsToSample(n);
        auto ready_data = SampleVectors(data_p, n, n_samples);

        if (verbose) {
            std::cout << "Clustering..." << std::endl;
        }

        for (int i = 0; i < _iters; ++i) {
            AssignAndUpdateCentroids(ready_data, n_samples);
            // split_clusters(iter_mat);
            if (alpha == dp) {
                PostprocessCentroids();
            }
            if (verbose)
                std::cout << "Iteration " << i + 1 << "/" << _iters
                          << " | Objective: " << std::endl;
        }

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
    std::vector<uint32_t>
    AssignAndUpdateCentroids(const vector_value_t* SKM_RESTRICT data, const size_t n) {
        // Data is a row-major matrix
        // Centroids is already in the PDX layout:
        //      Maybe we should use the PDXLayout abstraction and call Search() per vector
        //      Random access, no IVF... for now!
        // But, since collection is already rotated I need some hacks
        // Anyways...
        // fork::union
        // For each vector in data:
        //      1. size_t nn_idx = pdx.search(k=1, no_preprocessing, q)
        //      2. _cluster_sizes[nn_idx] += 1
        //      3. _tmp_centroids[[dim_positions_on_idx_layout]] += vector[:]
        // For each c_id: _tmp_centroids[[dim_positions_on_idx_layout]] /= _cluster_sizes[c_id]
        //      - Potentially also storing the norms for DP
        // Notes: I don't need proper assignments until the last iteration
    }

    std::vector<skmeans_centroid_value_t<q>> GetCentroids() const { return _centroids; }

    inline size_t GetNClusters() const { return _n_clusters; }

    inline bool IsTrained() const { return _trained; }

  protected:
    // Equidistant sampling similar to DuckDB's
    void SampleCentroids(vector_value_t* SKM_RESTRICT data, const size_t n) const {
        const auto jumps = static_cast<size_t>(std::floor(1.0 * n / _n_clusters));
        auto tmp_centroids_p = _tmp_centroids.data();
        for (size_t i = 0; i < n; i += jumps) {
            // TODO(@lkuffo, low): What if centroid scalar_t are not the same size of vector ones
            memcpy(tmp_centroids_p, data + i, sizeof(centroid_value_t) * _d);
            tmp_centroids_p += _d;
        }
        // Here, I may think of returning a PDX layout of the centroids
        if constexpr (std::is_same_v<Pruner, ADSamplingPruner<q>>) {
            // TODO(@lkuffo, high): Template bool for pruning or not
            std::vector<centroid_value_t> rotated_centroids(_n_clusters * _d);
            _pruner->Rotate(_tmp_centroids.data(), rotated_centroids.data(), _n_clusters);
            PDXLayout::PDXify<q, true>(
                rotated_centroids.data(), _centroids.data(), _n_clusters, _d
            );
        } else {
            PDXLayout::PDXify<q, true>(tmp_centroids_p.data(), _centroids.data(), _n_clusters, _d);
        }
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

    const uint32_t _iters;
    const size_t _n_clusters;
    const float _sampling_fraction;
    const size_t _d;
    bool _trained;
    bool verbose;
};
} // namespace skmeans

#endif // SUPERKMEANS_H
