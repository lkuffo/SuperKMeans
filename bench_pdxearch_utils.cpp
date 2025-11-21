#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include "include/superkmeans/common.h"

#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/pruners/adsampling.hpp"
#include "superkmeans/superkmeans.h"

// __attribute__((noinline))
void InitPositionsArray(
    size_t n_vectors,
    size_t& n_vectors_not_pruned,
    uint32_t* pruning_positions,
    float pruning_threshold,
    const float* pruning_distances
) {
    n_vectors_not_pruned = 0;
    for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
        pruning_positions[n_vectors_not_pruned] = vector_idx;
        n_vectors_not_pruned += pruning_distances[vector_idx] < pruning_threshold;
    }
};

// __attribute__((noinline))
void EvaluatePruningPredicateOnPositionsArray(
    const size_t n_vectors,
    size_t& n_vectors_not_pruned,
    uint32_t* pruning_positions,
    float pruning_threshold,
    float* pruning_distances
) {
    n_vectors_not_pruned = 0;
    for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
        pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
        n_vectors_not_pruned +=
            pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
    }
};

// __attribute__((noinline))
uint64_t EvaluatePruningPredicateOnMask(
    size_t n_vectors,
    size_t& n_vectors_not_pruned,
    float pruning_threshold,
    float* pruning_distances,
    uint64_t previous_mask
) {
    uint64_t mask = 0;
    for (size_t i = 0; i < n_vectors; ++i) {
        size_t v_idx = __builtin_ctzll(previous_mask);
        mask |= (uint64_t) (pruning_distances[v_idx] < pruning_threshold) << v_idx;
        previous_mask &= previous_mask - 1;
    }
    n_vectors_not_pruned = __builtin_popcountll(mask);
    return mask;
};

// __attribute__((noinline))
uint64_t InitPositionsMask(
    size_t n_vectors,
    size_t& n_vectors_not_pruned,
    float pruning_threshold,
    float* pruning_distances
) {
    uint64_t mask = 0;
    __builtin_assume(n_vectors <= 64);
    for (size_t i = 0; i < n_vectors; ++i) {
        mask |= (uint64_t) (pruning_distances[i] < pruning_threshold) << i;
    }
    n_vectors_not_pruned = __builtin_popcountll(mask);
    return mask;
};

// __attribute__((noinline))
void ArrayWork(
    const size_t n_vectors_not_pruned,
    float& tot,
    const float* distances_p,
    const uint32_t* pruning_positions
) {
    for (size_t j = 0; j < n_vectors_not_pruned; ++j) {
        //std::cout << " ->" << pruning_positions[j] << std::endl;
        //std::cout << " -->" << distances_p[pruning_positions[j]] << std::endl;
        tot += distances_p[pruning_positions[j]];
    }
}

// __attribute__((noinline))
void MaskWork(
    const size_t n_vectors_not_pruned,
    float& tot,
    const float* distances_p,
    uint64_t mask
) {
    for (size_t j = 0; j < n_vectors_not_pruned; ++j) {
        const size_t v_idx = __builtin_ctzll(mask);
        //std::cout << " ->" << v_idx << std::endl;
        //std::cout << " -->" << distances_p[v_idx] << std::endl;
        tot += distances_p[v_idx];
        mask &= mask - 1;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Compiles!" << std::endl;

    std::random_device rd;  // non-deterministic seed
    std::mt19937 gen(rd()); // Mersenne Twister
    std::uniform_real_distribution<float> dist(0.0f, 1.0);

    constexpr size_t n_vectors = 64;
    volatile size_t n_clusters = 32;
    volatile size_t sub_work = 8;
    float pruning_threshold = 0.03f;
    constexpr size_t epochs = 20;
    constexpr size_t iters = 200000;
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> distances(n_vectors * n_clusters);
    for (float& x : distances)
        x = dist(gen);

    ankerl::nanobench::Bench().epochs(epochs).epochIterations(iters).run("Mask", [&]() {
        uint64_t mask = 0;
        float* distances_p = distances.data();
        for (size_t i = 0; i < n_clusters; ++i) {
            size_t n_vectors_not_pruned = 0;
            mask =
                InitPositionsMask(n_vectors, n_vectors_not_pruned, pruning_threshold, distances_p);
            float tot = 0.0f;
            // std::cout << "mask " << n_vectors_not_pruned << std::endl;
            for (size_t z = 0; z < sub_work; ++z) {
                MaskWork(n_vectors_not_pruned, tot, distances_p, mask);
                uint64_t prev_mask = mask;
                mask = EvaluatePruningPredicateOnMask(
                    n_vectors_not_pruned, n_vectors_not_pruned, pruning_threshold, distances_p, prev_mask
                );
            }
            // std::cout << "tot " << tot << std::endl;
            distances_p += n_vectors;
        }
    });

    ankerl::nanobench::Bench().epochs(epochs).epochIterations(iters).run("Array", [&]() {
        alignas(64) uint32_t pruning_positions[64];
        float* distances_p = distances.data();
        std::iota(pruning_positions, pruning_positions + n_vectors, 0);
        for (size_t i = 0; i < n_clusters; ++i) {
            size_t n_vectors_not_pruned = 0;
            InitPositionsArray(
                n_vectors, n_vectors_not_pruned, pruning_positions, pruning_threshold, distances_p
            );
            float tot = 0.0f;
            // std::cout << "arr  " << n_vectors_not_pruned << std::endl;
            for (size_t z = 0; z < sub_work; ++z) {
                ArrayWork(n_vectors_not_pruned, tot, distances_p, pruning_positions);
                uint32_t prev_n_vectors_not_pruned = n_vectors_not_pruned;
                EvaluatePruningPredicateOnPositionsArray(
                    prev_n_vectors_not_pruned,
                    n_vectors_not_pruned,
                    pruning_positions,
                    pruning_threshold,
                    distances_p
                );
            }
            // std::cout << "tot " << tot << std::endl;
            distances_p += n_vectors;
        }
    });
}
