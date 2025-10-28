#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/pruners/adsampling.hpp"
#include "superkmeans/superkmeans.h"

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
    bool BENCHMARK_PDXIFY = false;
    bool BENCHMARK_ADSAMPLING = true;
    if (BENCHMARK_PDXIFY) {
        ankerl::nanobench::Bench().epochs(1).epochIterations(5000).run("PDXify[FULL]", [&]() {
            skmeans::PDXLayout<>::PDXify<true>(pdx_vec.data(), pdx_out.data(), n, d);
        });
        skmeans::PDXLayout<>::CheckBlockTranspose(pdx_vec.data(), pdx_out.data(), n, d);
        skmeans::PDXLayout<>::CheckBlockTranspose(pdx_vec.data(), pdx_out.data(), n, d);
        ankerl::nanobench::Bench().epochs(1).epochIterations(5000).run("PDXify", [&]() {
            skmeans::PDXLayout<>::PDXify<false>(pdx_vec.data(), pdx_out.data(), n, d);
        });
        skmeans::PDXLayout<>::CheckBlockTransposeNonFull(pdx_vec.data(), pdx_out.data(), n, d, 576, 192);
        // std::cout << "Distance: " << distance << std::endl;
    }

    // Rotation
    n = 65536 * 10;
    d = 512;
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> raw_in(d * n);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> rotated_out(d * n);
    for (float& x : raw_in)
        x = dist(gen);
    for (float& x : rotated_out)
        x = dist(gen);
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

}
