#ifndef SKMEANS_BOND_SEARCH_HPP
#define SKMEANS_BOND_SEARCH_HPP

#include <queue>

namespace skmeans {

template <Quantization q = f32>
class BondPruner {
    using DISTANCES_TYPE = skmeans_distance_t<q>;
    using DATA_TYPE = skmeans_value_t<q>;
    using CLUSTER_TYPE = Cluster<q>;
    using KNNCandidate_t = KNNCandidate<q>;
    using VectorComparator_t = VectorComparator<q>;

  public:
    uint32_t num_dimensions;

    BondPruner(uint32_t num_dimensions) : num_dimensions(num_dimensions) {};

    // TODO: Do not copy
    void PreprocessQuery(float* raw_query, float* query) {
        memcpy((void*) query, (void*) raw_query, num_dimensions * sizeof(DATA_TYPE));
    }

    template <Quantization Q = q>
    skmeans_distance_t<Q> GetPruningThreshold(
        uint32_t k,
        std::priority_queue<KNNCandidate<Q>, std::vector<KNNCandidate<Q>>, VectorComparator<Q>>&
            heap,
        const uint32_t current_dimension_idx
    ) {
        return heap.size() == k ? heap.top().distance
                                : std::numeric_limits<skmeans_distance_t<Q>>::max();
    }
};

} // namespace skmeans

#endif // SKMEANS_BOND_SEARCH_HPP
