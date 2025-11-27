#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/superkmeans.h"


std::vector<float> make_blobs(
    size_t n_samples,
    size_t n_features,
    size_t n_centers,
    unsigned int random_state = 1)
{
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

int main(int argc, char* argv[]) {
    std::cout << "Compiles!" << std::endl;
    constexpr size_t DIM = 128;
    std::vector<float> vec_a(DIM);
    std::vector<float> vec_b(DIM);

    std::random_device rd;  // non-deterministic seed
    std::mt19937 gen(rd()); // Mersenne Twister
    std::uniform_real_distribution<float> dist(0.0f, 1.0);
    for (float& x : vec_a)
        x = dist(gen);
    for (float& x : vec_b)
        x = dist(gen);

    // skmeans::skmeans_distance_t<skmeans::f32> distance = 0.0;
    // ankerl::nanobench::Bench().minEpochIterations(500000).run("d=128 | f32 | l2", [&]() {
    //     ankerl::nanobench::doNotOptimizeAway(
    //         distance = skmeans::DistanceComputer<skmeans::l2,
    //         skmeans::f32>::Horizontal(vec_a.data(), vec_b.data(), DIM)
    //     );
    // });
    // std::cout << "Distance: " << distance << std::endl;

    size_t n = 4096;
    size_t d = 768;
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> pdx_vec(d * n);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> pdx_out(d * n);
    for (float& x : pdx_vec)
        x = dist(gen);
    for (float& x : pdx_out)
        x = dist(gen);
    // .numWarmupIters()
    // https://nanobench.ankerl.com/reference.html#_CPPv4N6ankerl9nanobench5Bench15epochIterationsE8uint64_t

    // PDXifying
    bool BENCHMARK_PDXIFY = true;
    bool BENCHMARK_ADSAMPLING = false;

    // Rotation
    n = 65536 * 1;
    d = 512;
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> raw_in(d * n);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> rotated_out(d * n);
    if (BENCHMARK_ADSAMPLING) {
        skmeans::ADSamplingPruner<skmeans::f32> ads(d, 1.5);
        ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("RotationMatrixCreation", [&]() {
            ankerl::nanobench::doNotOptimizeAway(
                ads = skmeans::ADSamplingPruner<skmeans::f32>(d, 1.5)
            );
        });
        ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Rotate", [&]() {
            ads.Rotate(raw_in.data(), rotated_out.data(), n);
        });
    }

    // SKMeans
    n = 262144;
    d = 1024;
    size_t n_clusters = 1024;
    uint32_t n_iters = 1;
    float sampling_fraction = 1.0;
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> data = make_blobs(n, d, n_clusters);

    auto kmeans_state = skmeans::SuperKMeans<skmeans::f32, skmeans::l2>(n_clusters, d, n_iters, sampling_fraction, true);
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("SKMeans", [&]() {
        auto centroids = kmeans_state.Train(data.data(), n);
    });


}
