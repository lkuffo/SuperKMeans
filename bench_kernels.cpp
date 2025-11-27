#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/superkmeans.h"
#include <Eigen/Eigen/Dense>

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

    size_t n = 262144;
    // n = 65536;
    size_t centroids_n = 1024;
    size_t d = 1024;
    float distance = 0.0f;
    constexpr size_t THREADS = 14;
    omp_set_num_threads(THREADS);
    constexpr size_t EPOCHS = 5;
    constexpr size_t ITERATIONS = 5;
    const auto dims = skmeans::PDXLayout<skmeans::f32>::GetDimensionSplit(d);
    const size_t horizontal_d = dims.horizontal_d;
    const size_t vertical_d = dims.vertical_d;
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> data = make_blobs(n, d, 512);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> centroids = make_blobs(centroids_n, d, 512);
    std::cout << "Total distance calculations: " << n * centroids_n << std::endl;
    std::cout << "Vertical D: " << vertical_d << std::endl;
    std::cout << "Horizontal D: " << horizontal_d << std::endl;
    std::cout << "Eigen ISA: " << Eigen::SimdInstructionSetsInUse() << std::endl;
    std::cout << "Eigen # threads: " << Eigen::nbThreads() << " (note: it will always be 1 if BLAS is enabled)" << std::endl;

    using MatrixR = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixC = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;


    //
    // Eigen
    //
    std::vector<float> eigen_out_buf(n * centroids_n);
    Eigen::Map<const MatrixR> eigen_data(data.data(), n, d);
    Eigen::Map<MatrixR> eigen_centroids(centroids.data(), centroids_n, d);
    Eigen::Map<MatrixR> eigen_out(eigen_out_buf.data(), n, centroids_n);
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Eigen", [&]() {
        ankerl::nanobench::doNotOptimizeAway(
            eigen_out.noalias() = eigen_data * eigen_centroids.transpose()
        );
    });

    //
    // Eigen Batch L2sqr
    //
    Eigen::Map<const MatrixR> b_eigen_X(data.data(), n, d);
    Eigen::Map<MatrixR> b_eigen_Y(centroids.data(), centroids_n, d);
    const Eigen::VectorXf norms_x = b_eigen_X.rowwise().squaredNorm();
    const Eigen::VectorXf norms_y = b_eigen_Y.rowwise().squaredNorm();
    std::vector<uint32_t> out_knn(n);
    std::vector<float> out_distances(n);
    std::vector<float> all_distances(n * centroids_n);
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Batch_XRowMajor_YColMajor", [&]() {
        skmeans::BatchComputer<skmeans::l2, skmeans::f32>::Batch_XRowMajor_YColMajor(
            data.data(),
            centroids.data(),
            n,
            centroids_n,
            d,
            norms_x.data(),
            norms_y.data(),
            out_knn.data(),
            out_distances.data(),
            all_distances.data()
        );
    });


    Eigen::Map<MatrixR> eigen_data_t(data.data(), n, d);
    Eigen::Map<MatrixC> eigen_centroids_t(centroids.data(), d, centroids_n);
    Eigen::Map<MatrixC> eigen_out_t(eigen_out_buf.data(), n, centroids_n);
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Eigen (NT)", [&]() {
        ankerl::nanobench::doNotOptimizeAway(
            eigen_out_t.noalias() = eigen_data_t * eigen_centroids_t
        );
    });

    return 0;

    //
    // Horizontal Serial
    //
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Horizontal Serial", [&]() {
        auto data_p = data.data();
        // For each point
#pragma omp parallel for if (THREADS > 1) num_threads(THREADS)
        for (size_t i = 0; i < n; ++i) {
            // We query the centroids
            float local_distance = 0.0f;
            auto centroids_p = centroids.data();
            for (size_t j = 0; j < centroids_n; ++j) {
                if constexpr (THREADS > 1) {
                    ankerl::nanobench::doNotOptimizeAway(
                        local_distance = skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::Horizontal(
                            data_p, centroids_p, d
                        )
                    );
                } else {
                    ankerl::nanobench::doNotOptimizeAway(
                        distance = skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::Horizontal(
                            data_p, centroids_p, d
                        )
                    );
                }
                centroids_p += d;
            }
            data_p += d;
        }
    });

    //
    // Vertical Serial
    //
    size_t p_math = 0;
    std::vector<float> distances_vertical_serial(skmeans::VECTOR_CHUNK_SIZE);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> pdx_full_centroids(centroids_n * d);
    skmeans::PDXLayout<skmeans::f32>::PDXify<true>(
        centroids.data(), pdx_full_centroids.data(), centroids_n, d
    );
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Vertical Serial", [&]() {
        auto data_p = data.data();
        // For each point
#pragma omp parallel for if (THREADS > 1) num_threads(THREADS)
        for (size_t i = 0; i < n; ++i) {
            // We query the centroids
            std::vector<float> distances_vertical_serial_local;
            if constexpr (THREADS > 1) {
                distances_vertical_serial_local.resize(skmeans::VECTOR_CHUNK_SIZE);
            }
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j += skmeans::VECTOR_CHUNK_SIZE) {
                if constexpr (THREADS > 1) {
                    std::fill(
                        distances_vertical_serial_local.begin(),
                        distances_vertical_serial_local.end(), 0.0f
                    );
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlock(
                        data_p, centroids_p, 0, d, distances_vertical_serial_local.data()
                    );
                } else {
                    std::fill(
                        distances_vertical_serial.begin(), distances_vertical_serial.end(), 0.0f
                    );
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlock(
                        data_p, centroids_p, 0, d, distances_vertical_serial.data()
                    );
                }
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            if constexpr (THREADS == 1)
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
    skmeans::PDXLayout<skmeans::f32>::PDXify<false>(
        centroids.data(), pdx_centroids.data(), centroids_n, d
    );
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("PDX Serial", [&]() {
        auto data_p = data.data();
        // For each point
#pragma omp parallel for if (THREADS > 1) num_threads(THREADS)
        for (size_t i = 0; i < n; ++i) {
            std::vector<float> distances_local_full;
            if constexpr (THREADS > 1) {
                distances_local_full.reserve(skmeans::VECTOR_CHUNK_SIZE);
            }

            // We query the centroids
            auto centroids_p = pdx_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j += skmeans::VECTOR_CHUNK_SIZE) {
                if constexpr (THREADS > 1) {
                    std::fill(distances_local_full.begin(), distances_local_full.end(), 0.0f);
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlock(
                        data_p, centroids_p, 0, vertical_d, distances_local_full.data()
                    );
                } else {
                    std::fill(distances_pdx.begin(), distances_pdx.end(), 0.0f);
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlock(
                        data_p, centroids_p, 0, vertical_d, distances_pdx.data()
                    );
                }
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * vertical_d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * vertical_d;
                for (size_t k = vertical_d; k < d;
                     k += skmeans::H_DIM_SIZE) { // Go through the horizontal segments
                    for (size_t t = 0; t < skmeans::VECTOR_CHUNK_SIZE; ++t) { // Of 64 vectors
                        if constexpr (THREADS > 1) {
                            ankerl::nanobench::doNotOptimizeAway(
                                distances_local_full[t] +=
                                skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::Horizontal(
                                    data_p + k, centroids_p, skmeans::H_DIM_SIZE
                                )
                            );
                        } else {
                            ankerl::nanobench::doNotOptimizeAway(
                                distances_pdx[t] +=
                                skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::Horizontal(
                                    data_p + k, centroids_p, skmeans::H_DIM_SIZE
                                )
                            );
                        }
                        centroids_p += skmeans::H_DIM_SIZE;
                        p_math += skmeans::H_DIM_SIZE;
                    }
                }
            }
            if constexpr (THREADS == 1)
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
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Vertical Batched 64", [&]() {
        auto data_p = pdx_full_data.data();
        // For each point
#pragma omp parallel for if (THREADS > 1) num_threads(THREADS)
        for (size_t i = 0; i < n; i += BATCH_SIZE_64) {
            std::vector<float> distances_batch_vertical_local;
            if constexpr (THREADS > 1) {
                distances_batch_vertical_local.reserve(BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE);
            }
            // We query the centroids
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j += skmeans::VECTOR_CHUNK_SIZE) {
                if constexpr (THREADS > 1) {
                    std::fill(
                        distances_batch_vertical_local.begin(),
                        distances_batch_vertical_local.end(), 0.0f
                    );
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64(
                        data_p, centroids_p, 0, d, distances_batch_vertical_local.data()
                    );
                } else {
                    std::fill(
                        distances_batch_vertical.begin(), distances_batch_vertical.end(), 0.0f
                    );
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64(
                        data_p, centroids_p, 0, d, distances_batch_vertical.data()
                    );
                }
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            if constexpr (THREADS == 1)
                assert(p_math == centroids_n * d);
            data_p += BATCH_SIZE_64 * d;
            data_p_math += BATCH_SIZE_64 * d;
        }
        if constexpr (THREADS == 1)
            assert(data_p_math == n * d);
    });

    //
    //
    // Vertical Batched 64 V2
    //
    //
    p_math = 0;
    data_p_math = 0;
    std::vector<float> distances_batch_vertical_v2(BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE);
    std::fill(distances_batch_vertical_v2.begin(), distances_batch_vertical_v2.end(), 0.0f);
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Vertical Batched 64 v2", [&]() {
        auto data_p = pdx_full_data.data();
        // For each point
#pragma omp parallel for if (THREADS > 1) num_threads(THREADS)
        for (size_t i = 0; i < n; i += BATCH_SIZE_64) {
            std::vector<float> distances_batch_vertical_v2_local;
            if constexpr (THREADS > 1) {
                distances_batch_vertical_v2_local.reserve(
                    BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE
                );
            }

            // We query the centroids
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j += skmeans::VECTOR_CHUNK_SIZE) {
                if constexpr (THREADS > 1) {
                    std::fill(
                        distances_batch_vertical_v2_local.begin(),
                        distances_batch_vertical_v2_local.end(), 0.0f
                    );
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64V2(
                        data_p, centroids_p, 0, d,
                        // distances_batch_vertical_v2.data()
                        distances_batch_vertical_v2_local.data()
                    );
                } else {
                    std::fill(
                        distances_batch_vertical_v2.begin(), distances_batch_vertical_v2.end(), 0.0f
                    );
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64V2(
                        data_p, centroids_p, 0, d, distances_batch_vertical_v2.data()
                    );
                }

                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            if constexpr (THREADS == 1)
                assert(p_math == centroids_n * d);
            data_p += BATCH_SIZE_64 * d;
            data_p_math += BATCH_SIZE_64 * d;
        }
        if constexpr (THREADS == 1)
            assert(data_p_math == n * d);
    });

    //
    //
    // Vertical Batched 64 SIMD
    //
    //
    p_math = 0;
    data_p_math = 0;
    std::vector<float> distances_batch_vertical_64_simd(BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE);
    std::fill(
        distances_batch_vertical_64_simd.begin(), distances_batch_vertical_64_simd.end(), 0.0f
    );
    ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Vertical Batched 64 SIMD", [&]() {
        auto data_p = pdx_full_data.data();
        // For each point
#pragma omp parallel for if (THREADS > 1) num_threads(THREADS)
        for (size_t i = 0; i < n; i += BATCH_SIZE_64) {
            std::vector<float> distances_batch_vertical_64_simd_local;
            if constexpr (THREADS > 1) {
                distances_batch_vertical_64_simd_local.resize(
                    BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE
                );
            }
            // We query the centroids
            auto centroids_p = pdx_full_centroids.data();
            p_math = 0;
            for (size_t j = 0; j < centroids_n; j += skmeans::VECTOR_CHUNK_SIZE) {
                if constexpr (THREADS > 1) {
                    std::fill(
                        distances_batch_vertical_64_simd_local.begin(),
                        distances_batch_vertical_64_simd_local.end(), 0.0f
                    );
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64SIMD(
                        data_p, centroids_p, 0, d, distances_batch_vertical_64_simd_local.data()
                    );
                } else {
                    std::fill(
                        distances_batch_vertical_64_simd.begin(),
                        distances_batch_vertical_64_simd.end(), 0.0f
                    );
                    skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch64SIMD(
                        data_p, centroids_p, 0, d, distances_batch_vertical_64_simd.data()
                    );
                }
                centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
                p_math += skmeans::VECTOR_CHUNK_SIZE * d;
            }
            if constexpr (THREADS == 1)
                assert(p_math == centroids_n * d);
            data_p += BATCH_SIZE_64 * d;
            data_p_math += BATCH_SIZE_64 * d;
        }
        if constexpr (THREADS == 1)
            assert(data_p_math == n * d);
    });

    //
    // Vertical Batched 8 V2
    //
    // p_math = 0;
    // data_p_math = 0;
    // constexpr size_t BATCH_SIZE_8 = 8;
    // std::vector<float> distances_batch_vertical_8_v2(BATCH_SIZE_8 * skmeans::VECTOR_CHUNK_SIZE);
    // std::fill(distances_batch_vertical_8_v2.begin(), distances_batch_vertical_8_v2.end(), 0.0f);
    // skmeans::PDXLayout<skmeans::f32>::PDXify<true, BATCH_SIZE_8>(data.data(),
    // pdx_full_data.data(), n, d);
    // ankerl::nanobench::Bench().epochs(EPOCHS).epochIterations(ITERATIONS).run("Vertical Batched 8 v2", [&]() {
    //     auto data_p = pdx_full_data.data();
    //     // For each point
    //     for (size_t i = 0; i < n; i+=BATCH_SIZE_8) {
    //         // We query the centroids
    //         auto centroids_p = pdx_full_centroids.data();
    //         p_math = 0;
    //         for (size_t j = 0; j < centroids_n; j+=skmeans::VECTOR_CHUNK_SIZE) {
    //             std::fill(distances_batch_vertical_8_v2.begin(),
    //             distances_batch_vertical_8_v2.end(), 0.0f);
    //             skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::VerticalBlockBatch8V2(
    //                 data_p, centroids_p, 0, d, distances_batch_vertical_v2.data()
    //             );
    //             centroids_p += skmeans::VECTOR_CHUNK_SIZE * d;
    //             p_math += skmeans::VECTOR_CHUNK_SIZE * d;
    //         }
    //         assert(p_math == centroids_n * d);
    //         data_p += BATCH_SIZE_8 * d;
    //         data_p_math += BATCH_SIZE_8 * d;
    //     }
    //     assert(data_p_math == n * d);
    // });

    if constexpr (THREADS == 1) {
        std::cout << std::endl;
        std::cout << "Dist H: " << distance << std::endl;
        std::cout << "Dist PDX FULL: " << distances_vertical_serial[63] << std::endl;
        std::cout << "Dist PDX: " << distances_pdx[63] << std::endl;
        std::cout << "Dist PDX FULL BATCH 64: "
                  << distances_batch_vertical[BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE - 1]
                  << std::endl;
        std::cout << "Dist PDX FULL BATCH SIMD 64: "
                  << distances_batch_vertical_64_simd[BATCH_SIZE_64 * skmeans::VECTOR_CHUNK_SIZE - 1]
                  << std::endl;
        assert(almost_equal(distance, distances_vertical_serial[63]));
        assert(almost_equal(distance, distances_pdx[63]));
    }

}
