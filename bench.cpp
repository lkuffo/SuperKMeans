#define ANKERL_NANOBENCH_IMPLEMENT

#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/superkmeans.h"
#include "superkmeans/nanobench.h"

int main(int argc, char *argv[]) {
    std::cout << "Compiles!" << std::endl;
    constexpr size_t DIM = 128;
    std::vector<float> vec_a(DIM);
    std::vector<float> vec_b(DIM);

    std::random_device rd;   // non-deterministic seed
    std::mt19937 gen(rd());  // Mersenne Twister
    std::uniform_real_distribution<float> dist(0.0f, 1.0);
    for (float &x : vec_a) x = dist(gen);
    for (float &x : vec_b) x = dist(gen);


    // skmeans::skmeans_distance_t<skmeans::f32> distance = 0.0;
    // ankerl::nanobench::Bench().minEpochIterations(500000).run("d=128 | f32 | l2", [&]() {
    //     ankerl::nanobench::doNotOptimizeAway(
    //         distance = skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::Horizontal(vec_a.data(), vec_b.data(), DIM)
    //     );
    // });
    // std::cout << "Distance: " << distance << std::endl;

    volatile size_t d = 768;
    volatile size_t n = 4096;
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> pdx_vec(d * n);
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> pdx_out(d * n);
    ankerl::nanobench::Bench().minEpochIterations(5000).run("PDXify", [&]() {
        skmeans::PDXify<skmeans::f32, true>(pdx_vec.data(), pdx_out.data(), n, d);
    });
    // std::cout << "Distance: " << distance << std::endl;

}
