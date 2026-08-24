#pragma once

#include "superkmeans/common.h"
#include <vector>

namespace skmeans {

/*
 * PDX index structure for IVF
 */
template <Quantization q>
class IndexPDXIVF {
  public:
    using CLUSTER_TYPE = Cluster<q>;

    uint32_t num_dimensions{};
    uint32_t num_clusters{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<CLUSTER_TYPE> clusters;
};

template <>
class IndexPDXIVF<Quantization::sq8> {
  public:
    using CLUSTER_TYPE = Cluster<Quantization::sq8>;

    uint32_t num_dimensions{};
    uint32_t num_clusters{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<Cluster<Quantization::sq8>> clusters;

    float for_base{};
    float scale_factor{};
    float quantization_scale_squared{};
    float inverse_scale_factor_squared{};
};

template <>
class IndexPDXIVF<Quantization::sq4> {
  public:
    using CLUSTER_TYPE = Cluster<Quantization::sq4>;

    uint32_t num_dimensions{};
    uint32_t num_clusters{};
    uint32_t num_horizontal_dimensions{};
    uint32_t num_vertical_dimensions{};
    std::vector<Cluster<Quantization::sq4>> clusters;

    float quantization_scale_squared{};
    float inverse_scale_factor_squared{};
};

} // namespace skmeans
