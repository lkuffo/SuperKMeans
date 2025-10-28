#ifndef SKMEANS_IVF_HPP
#define SKMEANS_IVF_HPP

#include "superkmeans/common.h"
#include "superkmeans/pdx/utils.h"
#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace skmeans {

/******************************************************************
 * Very rudimentary memory to IVF index reader
 ******************************************************************/
template <Quantization q>
class IndexPDXIVF {};

template <>
class IndexPDXIVF<f32> {
  public:
    using CLUSTER_TYPE = Cluster<f32>;

    std::unique_ptr<char[]> file_buffer;

    uint32_t num_dimensions{};
    uint32_t num_clusters{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<CLUSTER_TYPE> clusters;
    float* means{};
    bool is_ivf{};
    bool is_normalized{};
    float* centroids{};
    float* centroids_pdx{};

    void Restore(const std::string& filename) {
        file_buffer = MmapFile(filename);
        Load(file_buffer.get());
    }

    void Load(char* input) {
        char* next_value = input;
        num_dimensions = ((uint32_t*) input)[0];
        num_vertical_dimensions = ((uint32_t*) input)[1];
        num_horizontal_dimensions = ((uint32_t*) input)[2];

        next_value += sizeof(uint32_t) * 3;
        num_clusters = ((uint32_t*) next_value)[0];
        next_value += sizeof(uint32_t);
        auto* nums_embeddings = (uint32_t*) next_value;
        next_value += num_clusters * sizeof(uint32_t);
        clusters.resize(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            CLUSTER_TYPE& cluster = clusters[i];
            cluster.num_embeddings = nums_embeddings[i];
            cluster.data = (float*) next_value;
            next_value += sizeof(float) * cluster.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            CLUSTER_TYPE& cluster = clusters[i];
            cluster.indices = (uint32_t*) next_value;
            next_value += sizeof(uint32_t) * cluster.num_embeddings;
        }
        means = (float*) next_value;
        next_value += sizeof(float) * num_dimensions;
        is_normalized = ((char*) next_value)[0];
        next_value += sizeof(char);
        is_ivf = ((char*) next_value)[0];
        next_value += sizeof(char);
        if (is_ivf) {
            centroids = (float*) next_value;
            next_value += sizeof(float) * num_clusters * num_dimensions;
            centroids_pdx = (float*) next_value;
        }
    }
};

template <>
class IndexPDXIVF<u8> {
  public:
    using CLUSTER_TYPE = Cluster<u8>;

    std::unique_ptr<char[]> file_buffer;

    uint32_t num_dimensions{};
    uint32_t num_clusters{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<Cluster<u8>> clusters;
    float* means{};
    bool is_ivf{};
    bool is_normalized{};
    float* centroids{};
    float* centroids_pdx{};

    float for_base{};
    float scale_factor{};

    void Restore(const std::string& filename) {
        file_buffer = MmapFile(filename);
        Load(file_buffer.get());
    }

    void Load(char* input) {
        char* next_value = input;
        num_dimensions = ((uint32_t*) input)[0];
        num_vertical_dimensions = ((uint32_t*) input)[1];
        num_horizontal_dimensions = ((uint32_t*) input)[2];

        next_value += sizeof(uint32_t) * 3;
        num_clusters = ((uint32_t*) next_value)[0];
        next_value += sizeof(uint32_t);
        auto* nums_embeddings = (uint32_t*) next_value;
        next_value += num_clusters * sizeof(uint32_t);
        clusters.resize(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            CLUSTER_TYPE& cluster = clusters[i];
            cluster.num_embeddings = nums_embeddings[i];
            cluster.data = (uint8_t*) next_value;
            next_value += sizeof(uint8_t) * cluster.num_embeddings * num_dimensions;
        }
        for (size_t i = 0; i < num_clusters; ++i) {
            CLUSTER_TYPE& cluster = clusters[i];
            cluster.indices = (uint32_t*) next_value;
            next_value += sizeof(uint32_t) * cluster.num_embeddings;
        }
        // means = (float *) next_value;
        // next_value += sizeof(float) * num_dimensions;
        is_normalized = ((char*) next_value)[0];
        next_value += sizeof(char);
        is_ivf = ((char*) next_value)[0];
        next_value += sizeof(char);
        if (is_ivf) {
            centroids = (float*) next_value;
            next_value += sizeof(float) * num_clusters * num_dimensions;
            centroids_pdx = (float*) next_value;
        }

        for_base = ((float*) next_value)[0];
        next_value += sizeof(float);
        scale_factor = ((float*) next_value)[0];
        next_value += sizeof(float);
    }
};

} // namespace skmeans

#endif // SKMEANS_IVF_HPP
