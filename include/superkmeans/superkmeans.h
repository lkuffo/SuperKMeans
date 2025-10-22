#pragma once
#ifndef SUPERKMEANS_H
#define SUPERKMEANS_H

#include "common.h"
#include "distance_computers/base_computers.hpp"

namespace skmeans {
template <Quantization q = f32, DistanceFunction alpha = l2, class Pruner = ADSamplingPruner<q>>
class SuperKMeans {
    using centroid_value_t = skmeans_centroid_value_t<q>;
    using vector_value_t   = skmeans_value_t<q>;

  public:
    SuperKMeans(size_t n_clusters, uint32_t iters = 25, float sampling_fraction = 0.50)
        : _iters(iters), _n_clusters(n_clusters), _sampling_fraction(sampling_fraction),
          _trained(false) {
        SKMEANS_ENSURE_POSITIVE(n_clusters);
        SKMEANS_ENSURE_POSITIVE(iters);
        SKMEANS_ENSURE_POSITIVE(sampling_fraction);
        if (sampling_fraction > 1.0) {
            throw std::invalid_argument("sampling_fraction must be smaller than 1");
        }
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
    std::vector<skmeans_centroid_value_t<q>> train(const vector_value_t* SKM_RESTRICT data,
                                                   const size_t n, const size_t d,
                                                   uint32_t   num_threads = -1,
                                                   const bool verbose     = false) {
        SKMEANS_ENSURE_POSITIVE(n);
        SKMEANS_ENSURE_POSITIVE(d);
        if (_trained) {
            throw std::runtime_error("The clustering has already been trained");
        }

        if (n < _n_clusters) {
            throw std::runtime_error(
                "The number of points should be at least as large as the number of clusters");
        }

        vector_value_t* SKM_RESTRICT data_p = data;
        /*
         * 1. Sample using sampling_fraction. In this case you need to create a new buffer for sure.
         * 2. If metric is dp, normalize the vectors so you can use l2 always
         * 3. Transform data in data_p with Pruner preprocessing
         *    - ADSampling: Returns pointer to new buffer
         *    - Flat      : Zero copy (pass the pointer back)
         * 4. Sample centroids from data, copy them in the centroid buffer
         * 5. PDXify collection (must do multiprocessing, and must go into new buffer)
         * 6. Loop: assign+update, split
         * 7. Finalize:
         *    - I need to unrotate the centroids (Do assignments change if I derotate?)
         *    - I am not sure if I should just return the assignments
         *    - Assigning later would mean that I need to do the rotation also later
         */

        if (verbose) {
            std::cout << "Clustering..." << std::endl;
        }

        for (int i = 0; i < _iters; ++i) {
            assign_clusters(iter_mat, iter_norms, num_threads);
            update_centroids(iter_mat);
            split_clusters(iter_mat);
            postprocess_centroids();
            if (verbose)
                std::cout << "Iteration " << i + 1 << "/" << _iters
                          << " | Objective: " << cost(iter_mat) << std::endl;
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
    std::vector<uint32_t> assign(const vector_value_t* SKM_RESTRICT data, const size_t n) {
        SKMEANS_ENSURE_POSITIVE(n);
    }

    std::vector<skmeans_centroid_value_t<q>> get_centroids() const { return _centroids; }

    inline size_t get_n_clusters() const { return _n_clusters; }

    inline bool is_trained() const { return _trained; }

  protected:
    std::vector<centroid_value_t> _centroids;
    std::vector<uint32_t>         _assignments;
    std::vector<uint32_t>         _cluster_sizes;

    const uint32_t _iters;
    const size_t   _n_clusters;
    const float    _sampling_fraction;
    bool           _trained;
};
} // namespace skmeans

#endif // SUPERKMEANS_H
