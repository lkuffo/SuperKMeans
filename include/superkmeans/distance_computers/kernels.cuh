#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include <Eigen/Eigen/Dense>

#include "superkmeans/common.h"
// #include "superkmeans/distance_computers/base_computers.h"
// #include "superkmeans/pdx/layout.h"
// #include "superkmeans/profiler.h"



namespace skmeans {
namespace kernels {

using distance_t = skmeans_distance_t<Quantization::f32>;
using data_t = skmeans_value_t<Quantization::f32>;
using norms_t = skmeans_value_t<Quantization::f32>;

void first_blas(
    const int batch_n_x,
    const int batch_n_y,
    const int i,
    const int j,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    distance_t* out_distances,
    uint32_t* out_knn,
    const cudaStream_t stream
);

void norms(
    const int batch_n_x,
    const int batch_n_y,
    const int i,
    const int j,
    const norms_t* norms_x,
    const norms_t* norms_y,
    float* all_distances_buffer,
    const cudaStream_t stream
);

}
}
