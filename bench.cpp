#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/superkmeans.h"

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

    skmeans::skmeans_distance_t<skmeans::f32> distance =
        skmeans::DistanceComputer<skmeans::l2, skmeans::f32>::Horizontal(vec_a.data(), vec_b.data(), DIM);

    std::cout << "Distance: " << distance << std::endl;
}
