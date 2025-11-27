#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include "include/superkmeans/common.h"

#include <iostream>
#include <random>
#include <vector>

#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/superkmeans.h"

size_t InitPositionsIf(
    size_t n_vectors,
    uint32_t* pruning_positions,
    float pruning_threshold,
    const float* pruning_distances
) {
    size_t n_vectors_not_pruned = 0;
    for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
        if (pruning_distances[vector_idx] < pruning_threshold) {
            pruning_positions[n_vectors_not_pruned] = vector_idx;
            n_vectors_not_pruned++;
        }
    }
    return n_vectors_not_pruned;
};

// __attribute__((noinline))
uint32_t InitPositionsArray(
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
    return n_vectors_not_pruned;
};

// __attribute__((noinline))
size_t EvaluatePruningPredicateOnPositionsArray(
    const size_t n_vectors,
    size_t& n_vectors_not_pruned,
    uint32_t* pruning_positions,
    float pruning_threshold,
    const float* pruning_distances
) {
    n_vectors_not_pruned = 0;
    for (size_t vector_idx = 0; vector_idx < n_vectors; ++vector_idx) {
        pruning_positions[n_vectors_not_pruned] = pruning_positions[vector_idx];
        n_vectors_not_pruned +=
            pruning_distances[pruning_positions[vector_idx]] < pruning_threshold;
    }
    return n_vectors_not_pruned;
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
float ArrayWork(
    const size_t n_vectors_not_pruned,
    float& tot,
    const float* distances_p,
    const uint32_t* pruning_positions
) {
    for (size_t j = 0; j < n_vectors_not_pruned; ++j) {
        // std::cout << " ->" << pruning_positions[j] << std::endl;
        // std::cout << " -->" << distances_p[pruning_positions[j]] << std::endl;
        tot += distances_p[pruning_positions[j]];
    }
    return tot;
}

size_t EvaluatePruningPredicateOnIf(
    size_t n_vectors,
    size_t& n_vectors_not_pruned,
    float pruning_threshold,
    const float* pruning_distances
) {
    n_vectors_not_pruned = 0;
    for (size_t i = 0; i < n_vectors; ++i) {
        if (pruning_distances[i] < pruning_threshold) {
            n_vectors_not_pruned += 1;
        }
    }
    return n_vectors_not_pruned;
};

float IfWork(
    const size_t n_vectors,
    float& tot,
    float pruning_threshold,
    const float* distances_p
) {
    for (size_t j = 0; j < n_vectors; ++j) {
        if (distances_p[j] < pruning_threshold) {
            tot += distances_p[j];
        }
    }
    return tot;
}

// __attribute__((noinline))
float MaskWork(
    const size_t n_vectors_not_pruned,
    float& tot,
    const float* distances_p,
    uint64_t mask
) {
    for (size_t j = 0; j < n_vectors_not_pruned; ++j) {
        const size_t v_idx = __builtin_ctzll(mask);
        tot += distances_p[v_idx];
        mask &= mask - 1;
    }
    return tot;
}

int main(int argc, char* argv[]) {
    std::cout << "Compiles!" << std::endl;

    std::random_device rd;  // non-deterministic seed
    std::mt19937 gen(rd()); // Mersenne Twister
    std::uniform_real_distribution<float> dist(0.0f, 1.0);

    constexpr size_t n_vectors = 64;
    volatile size_t n_clusters = 32;
    volatile size_t sub_work = 8;
    constexpr size_t epochs = 20;
    constexpr size_t iters = 20000;
    std::vector<skmeans::skmeans_value_t<skmeans::f32>> distances(n_vectors * n_clusters);
    for (float& x : distances)
        x = dist(gen);
    std::vector<float> pruning_thresholds = {0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 0.8f};
    // for (float& pruning_threshold : pruning_thresholds) {
    //
    //     ankerl::nanobench::Bench().epochs(epochs).epochIterations(iters).run(
    //         "If INIT  (" + std::to_string(pruning_threshold) + ")",
    //         [&]() {
    //             alignas(64) uint32_t pruning_positions[64];
    //             float* distances_p = distances.data();
    //             for (size_t i = 0; i < n_clusters; ++i) {
    //                 size_t _x;
    //                 ankerl::nanobench::doNotOptimizeAway(
    //                         _x = InitPositionsIf(
    //                         n_vectors,
    //                         pruning_positions,
    //                         pruning_threshold,
    //                         distances_p
    //                     )
    //                 );
    //                 distances_p += n_vectors;
    //             }
    //         }
    //     );
    //
    //     ankerl::nanobench::Bench().epochs(epochs).epochIterations(iters).run(
    //         "Mask INIT (" + std::to_string(pruning_threshold) + ")",
    //         [&]() {
    //             uint64_t mask = 0;
    //             float* distances_p = distances.data();
    //             for (size_t i = 0; i < n_clusters; ++i) {
    //                 size_t n_vectors_not_pruned = 0;
    //                 ankerl::nanobench::doNotOptimizeAway(
    //                     mask = InitPositionsMask(
    //                         n_vectors, n_vectors_not_pruned, pruning_threshold, distances_p
    //                     )
    //                 );
    //                 distances_p += n_vectors;
    //             }
    //         }
    //     );
    //
    //     ankerl::nanobench::Bench().epochs(epochs).epochIterations(iters).run(
    //         "Array INIT(" + std::to_string(pruning_threshold) + ")",
    //         [&]() {
    //             alignas(64) uint32_t pruning_positions[64];
    //             float* distances_p = distances.data();
    //             std::iota(pruning_positions, pruning_positions + n_vectors, 0);
    //             for (size_t i = 0; i < n_clusters; ++i) {
    //                 size_t n_vectors_not_pruned = 0;
    //                 ankerl::nanobench::doNotOptimizeAway(
    //                         InitPositionsArray(
    //                         n_vectors,
    //                         n_vectors_not_pruned,
    //                         pruning_positions,
    //                         pruning_threshold,
    //                         distances_p
    //                     )
    //                 );
    //                 distances_p += n_vectors;
    //             }
    //         }
    //     );
    // }

    for (float& pruning_threshold : pruning_thresholds) {

        // ankerl::nanobench::Bench().epochs(epochs).epochIterations(iters).run(
        //     "If   (" + std::to_string(pruning_threshold) + ")",
        //     [&]() {
        //         alignas(64) uint32_t pruning_positions[64];
        //         float* distances_p = distances.data();
        //         for (size_t i = 0; i < n_clusters; ++i) {
        //             size_t _x;
        //             ankerl::nanobench::doNotOptimizeAway(
        //                 _x = InitPositionsIf(
        //                     n_vectors, pruning_positions, pruning_threshold, distances_p
        //                 )
        //             );
        //             float tot = 0.0f;
        //             size_t n_vectors_not_pruned = 0;
        //             for (size_t z = 0; z < sub_work; ++z) {
        //                 float t_;
        //                 size_t d2_;
        //                 ankerl::nanobench::doNotOptimizeAway(
        //                     t_ = IfWork(n_vectors, tot, pruning_threshold, distances_p)
        //                 );
        //                 ankerl::nanobench::doNotOptimizeAway(
        //                     d2_ = EvaluatePruningPredicateOnIf(
        //                         n_vectors, n_vectors_not_pruned, pruning_threshold, distances_p
        //                     )
        //                 );
        //             }
        //             distances_p += n_vectors;
        //         }
        //     }
        // );

        ankerl::nanobench::Bench().epochs(epochs).epochIterations(iters).run(
            "Mask (" + std::to_string(pruning_threshold) + ")",
            [&]() {
                uint64_t mask = 0;
                float* distances_p = distances.data();
                for (size_t i = 0; i < n_clusters; ++i) {
                    size_t n_vectors_not_pruned = 0;
                    ankerl::nanobench::doNotOptimizeAway(
                        mask = InitPositionsMask(
                            n_vectors, n_vectors_not_pruned, pruning_threshold, distances_p
                        )
                    );
                    float tot = 0.0f;
                    // std::cout << "mask " << n_vectors_not_pruned << std::endl;
                    for (size_t z = 0; z < sub_work; ++z) {
                        float t_;
                        ankerl::nanobench::doNotOptimizeAway(
                            t_ = MaskWork(n_vectors_not_pruned, tot, distances_p, mask)
                        );
                        ankerl::nanobench::doNotOptimizeAway(
                            t_ = MaskWork(n_vectors_not_pruned, tot, distances_p, mask)
                        );
                        uint64_t prev_mask = mask;
                        ankerl::nanobench::doNotOptimizeAway(
                            mask = EvaluatePruningPredicateOnMask(
                                n_vectors_not_pruned,
                                n_vectors_not_pruned,
                                pruning_threshold,
                                distances_p,
                                prev_mask
                            )
                        );
                    }
                    // std::cout << "tot " << tot << std::endl;
                    distances_p += n_vectors;
                }
            }
        );

        ankerl::nanobench::Bench().epochs(epochs).epochIterations(iters).run(
            "Array(" + std::to_string(pruning_threshold) + ")",
            [&]() {
                alignas(64) uint32_t pruning_positions[64];
                float* distances_p = distances.data();
                // std::iota(pruning_positions, pruning_positions + n_vectors, 0);
                for (size_t i = 0; i < n_clusters; ++i) {
                    size_t n_vectors_not_pruned = 0;
                    uint32_t d2_;
                    ankerl::nanobench::doNotOptimizeAway(
                        d2_ = InitPositionsArray(
                            n_vectors,
                            n_vectors_not_pruned,
                            pruning_positions,
                            pruning_threshold,
                            distances_p
                        )
                    );
                    float tot = 0.0f;
                    // std::cout << "arr  " << n_vectors_not_pruned << std::endl;
                    for (size_t z = 0; z < sub_work; ++z) {
                        float t_;
                        size_t d_;
                        ankerl::nanobench::doNotOptimizeAway(
                            t_ =
                                ArrayWork(n_vectors_not_pruned, tot, distances_p, pruning_positions)
                        );
                        ankerl::nanobench::doNotOptimizeAway(
                            t_ =
                                ArrayWork(n_vectors_not_pruned, tot, distances_p, pruning_positions)
                        );
                        uint32_t prev_n_vectors_not_pruned = n_vectors_not_pruned;
                        ankerl::nanobench::doNotOptimizeAway(
                            d_ = EvaluatePruningPredicateOnPositionsArray(
                                prev_n_vectors_not_pruned,
                                n_vectors_not_pruned,
                                pruning_positions,
                                pruning_threshold,
                                distances_p
                            )
                        );
                    }
                    // std::cout << "tot " << tot << std::endl;
                    distances_p += n_vectors;
                }
            }
        );
    }
}
