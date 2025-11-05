#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

#include <Eigen/Eigen/Dense>
#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/pruners/adsampling.hpp"
#include "superkmeans/superkmeans.h"

std::vector<float>
make_blobs(size_t n_samples, size_t n_features, size_t n_centers, unsigned int random_state = 1) {
    std::mt19937 gen(random_state);

    // Random cluster centers
    std::normal_distribution<float> center_dist(0.0f, 1.0f);
    std::vector<std::vector<float>> centers(n_centers, std::vector<float>(n_features));
    for (auto& c : centers)
        for (auto& x : c)
            x = center_dist(gen);

    // Distributions for choosing cluster and spreading points
    std::uniform_int_distribution<size_t> cluster_dist(0, n_centers - 1);
    std::normal_distribution<float> point_dist(0.0f, 1.0f);

    // Flattened result: row-major layout [sample0_dim0, sample0_dim1, ..., sampleN_dimD]
    std::vector<float> data;
    data.reserve(n_samples * n_features);

    for (size_t i = 0; i < n_samples; ++i) {
        const auto& center = centers[cluster_dist(gen)];
        for (size_t j = 0; j < n_features; ++j)
            data.push_back(center[j] + point_dist(gen));
    }

    return data;
}

bool almost_equal(float a, float b, float rel_tol = 1e-6f, float abs_tol = 1e-9f) {
    return std::fabs(a - b) <= std::max(rel_tol * std::max(std::fabs(a), std::fabs(b)), abs_tol);
}

int main(int argc, char* argv[]) {
    std::cout << "Compiles!" << std::endl;


    size_t n = 131072;
    size_t centroids_n = 1024;
    size_t d = 512;
    float distance = 0.0f;
    auto [horizontal_d, vertical_d] = skmeans::PDXLayout<skmeans::f32>::GetDimensionSplit(d);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> data = make_blobs(n, d, 512);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> centroids = make_blobs(centroids_n, d, 512);
    std::cout << "Total distance calculations: " << n * centroids_n << std::endl;
    std::cout << "Vertical D: " << vertical_d << std::endl;
    std::cout << "Horizontal D: " << horizontal_d << std::endl;

    //
    // Eigen
    //
    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixC = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    std::vector<float> eigen_out_buf(n * centroids_n);
    Eigen::Map<const MatrixR> eigen_data(data.data(), n, d);
    Eigen::Map<MatrixR> eigen_centroids(centroids.data(), centroids_n, d);
    Eigen::Map<MatrixR> eigen_out(eigen_out_buf.data(), n, centroids_n);

    omp_set_num_threads(1);
    // Eigen::setNbThreads(10);
    std::cout << "Eigen # threads: " << Eigen::nbThreads() << std::endl;
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Eigen", [&]() {
        ankerl::nanobench::doNotOptimizeAway(
        eigen_out.noalias() = eigen_data * eigen_centroids.transpose()
        );
    });

    //
    // Horizontal Serial
    //
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Horizontal Serial", [&]() {
        auto data_p = data.data();
        // For each point
        for (size_t i = 0; i < n; ++i) {
            // We query the centroids
            auto centroids_p = centroids.data();
            for (size_t j = 0; j < centroids_n; ++j) {
                ankerl::nanobench::doNotOptimizeAway(
                    distance = skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::Horizontal(
                        data_p, centroids_p, d
                    )
                );
                centroids_p += d;
            }
            data_p += d;
        }
    });

    //
    // Vertical Serial
    //
    size_t p_math = 0;
    std::vector<float> distances_full(skmeans::VECTOR_CHUNK_SIZE);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> pdx_full_centroids(centroids_n * d);
    skmeans::PDXLayout<skmeans::f32>::PDXify<true>(centroids.data(), pdx_full_centroids.data(), centroids_n, d);
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Vertical Serial", [&]() {
        auto data_p = data.data();
        // For each point
        for (size_t i = 0; i < n; ++i) {
            // We query the centroids
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j+=skmeans::VECTOR_CHUNK_SIZE) {
                std::fill(distances_full.begin(), distances_full.end(), 0.0f);
                skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlock(
                    data_p, centroids_p, 0, d, distances_full.data()
                );
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            assert(p_math == centroids_n * d);
            data_p += d;
        }
    });

    //
    // PDX Serial
    //
    std::vector<float> distances_pdx(skmeans::VECTOR_CHUNK_SIZE);
    std::fill(distances_pdx.begin(), distances_pdx.end(), 0);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> pdx_centroids(centroids_n * d);
    skmeans::PDXLayout<skmeans::f32>::PDXify<false>(centroids.data(), pdx_centroids.data(), centroids_n, d);
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("PDX Serial", [&]() {
        auto data_p = data.data();
        // For each point
        for (size_t i = 0; i < n; ++i) {
            // We query the centroids
            auto centroids_p = pdx_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j+=skmeans::VECTOR_CHUNK_SIZE) {
                std::fill(distances_pdx.begin(), distances_pdx.end(), 0.0f);
                skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlock(
                    data_p, centroids_p, 0, vertical_d, distances_pdx.data()
                );
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * vertical_d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * vertical_d;
                for (size_t k = vertical_d; k < d; k+=skmeans::H_DIM_SIZE) { // Go through the horizontal segments
                    for (size_t t = 0; t < skmeans::VECTOR_CHUNK_SIZE; ++t) { // Of 64 vectors
                        ankerl::nanobench::doNotOptimizeAway(
                            distances_pdx[t] += skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::Horizontal(
                                data_p + k, centroids_p, skmeans::H_DIM_SIZE
                            )
                        );
                        centroids_p += skmeans::H_DIM_SIZE;
                        p_math += skmeans::H_DIM_SIZE;
                    }
                }
            }
            assert(p_math == centroids_n * d);
            data_p += d;
        }
    });


    //
    // Vertical Batched 64
    //
    p_math = 0;
    size_t data_p_math = 0;
    constexpr size_t BATCH_SIZE_64 = 64;
    std::vector<float> distances_batch_vertical(BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> pdx_full_data(n * d);
    skmeans::PDXLayout<skmeans::f32>::PDXify<true>(data.data(), pdx_full_data.data(), n, d);
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Vertical Batched 64", [&]() {
        auto data_p = pdx_full_data.data();
        // For each point
        for (size_t i = 0; i < n; i+=BATCH_SIZE_64) {
            // We query the centroids
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j+=skmeans::VECTOR_CHUNK_SIZE) {
                std::fill(distances_batch_vertical.begin(), distances_batch_vertical.end(), 0.0f);
                skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64(
                    data_p, centroids_p, 0, d, distances_batch_vertical.data()
                );
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            assert(p_math == centroids_n * d);
            data_p += BATCH_SIZE_64 * d;
            data_p_math += BATCH_SIZE_64 * d;
        }
        assert(data_p_math == n * d);
    });

    //
    // Vertical Batched 64 V2
    //
    p_math = 0;
    data_p_math = 0;
    std::vector<float> distances_batch_vertical_v2(BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE);
    std::fill(distances_batch_vertical_v2.begin(), distances_batch_vertical_v2.end(), 0.0f);
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Vertical Batched 64 v2", [&]() {
        auto data_p = pdx_full_data.data();
        // For each point
        for (size_t i = 0; i < n; i+=BATCH_SIZE_64) {
            // We query the centroids
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j+=skmeans::VECTOR_CHUNK_SIZE) {
                std::fill(distances_batch_vertical_v2.begin(), distances_batch_vertical_v2.end(), 0.0f);
                skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64V2(
                    data_p, centroids_p, 0, d, distances_batch_vertical_v2.data()
                );
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            assert(p_math == centroids_n * d);
            data_p += BATCH_SIZE_64 * d;
            data_p_math += BATCH_SIZE_64 * d;
        }
        assert(data_p_math == n * d);
    });

    //
    // Vertical Batched 64 SIMD
    //
    p_math = 0;
    data_p_math = 0;
    std::vector<float> distances_batch_vertical_64_simd(BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE);
    std::fill(distances_batch_vertical_64_simd.begin(), distances_batch_vertical_64_simd.end(), 0.0f);
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Vertical Batched 64 SIMD", [&]() {
        auto data_p = pdx_full_data.data();
        // For each point
        for (size_t i = 0; i < n; i+=BATCH_SIZE_64) {
            // We query the centroids
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j+=skmeans::VECTOR_CHUNK_SIZE) {
                std::fill(distances_batch_vertical_64_simd.begin(), distances_batch_vertical_64_simd.end(), 0.0f);
                skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64SIMD(
                    data_p, centroids_p, 0, d, distances_batch_vertical_64_simd.data()
                );
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            assert(p_math == centroids_n * d);
            data_p += BATCH_SIZE_64 * d;
            data_p_math += BATCH_SIZE_64 * d;
        }
        assert(data_p_math == n * d);
    });

    //
    // Vertical Batched 8 V2
    //
    p_math = 0;
    data_p_math = 0;
    constexpr size_t BATCH_SIZE_8 = 8;
    std::vector<float> distances_batch_vertical_8_v2(BATCH_SIZE_8 * skmeans::VECTOR_CHUNK_SIZE);
    std::fill(distances_batch_vertical_8_v2.begin(), distances_batch_vertical_8_v2.end(), 0.0f);
    skmeans::PDXLayout<skmeans::f32>::PDXify<true, BATCH_SIZE_8>(data.data(), pdx_full_data.data(), n, d);
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Vertical Batched 8 v2", [&]() {
        auto data_p = pdx_full_data.data();
        // For each point
        for (size_t i = 0; i < n; i+=BATCH_SIZE_8) {
            // We query the centroids
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j+=skmeans::VECTOR_CHUNK_SIZE) {
                std::fill(distances_batch_vertical_8_v2.begin(), distances_batch_vertical_8_v2.end(), 0.0f);
                skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch8V2(
                    data_p, centroids_p, 0, d, distances_batch_vertical_v2.data()
                );
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            assert(p_math == centroids_n * d);
            data_p += BATCH_SIZE_8 * d;
            data_p_math += BATCH_SIZE_8 * d;
        }
        assert(data_p_math == n * d);
    });


    std::cout << std::endl;
    std::cout << "Dist H: " << distance << std::endl;
    std::cout << "Dist PDX FULL: " << distances_full[63] << std::endl;
    std::cout << "Dist PDX: " << distances_pdx[63] << std::endl;
    std::cout << "Dist PDX FULL BATCH 64: " << distances_batch_vertical[BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE - 1] << std::endl;
    std::cout << "Dist PDX FULL BATCH SIMD 64: " << distances_batch_vertical_64_simd[BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE - 1] << std::endl;
    assert(almost_equal(distance, distances_full[63]));
    assert(almost_equal(distance, distances_pdx[63]));
}
