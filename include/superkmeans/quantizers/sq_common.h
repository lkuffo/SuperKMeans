#pragma once

#include "superkmeans/common.h"

#include <algorithm>
#include <limits>
#include <omp.h>

namespace skmeans {

struct ScalarQuantizationParams {
    float quantization_base;
    float quantization_scale;
    float inv_quantization_scale;
};

inline ScalarQuantizationParams ComputeScalarQuantizationParams(
    const float* embeddings,
    size_t total_elements,
    float max_value
) {
    float global_min = std::numeric_limits<float>::max();
    float global_max = std::numeric_limits<float>::lowest();

#pragma omp parallel for reduction(min : global_min) reduction(max : global_max)                   \
    num_threads(g_n_threads)
    for (size_t i = 0; i < total_elements; ++i) {
        global_min = std::min(global_min, embeddings[i]);
        global_max = std::max(global_max, embeddings[i]);
    }

    const float range = global_max - global_min;
    const float scale = (range > 0) ? max_value / range : 1.0f;
    return {global_min, scale, 1.0f / scale};
}

} // namespace skmeans
