#pragma once
#ifndef SUPERKMEANS_BASE_COMPUTERS_HPP
#define SUPERKMEANS_BASE_COMPUTERS_HPP

#include "superkmeans/common.h"

#ifdef __ARM_NEON
#include "neon_computers.hpp"
#endif

#if defined(__AVX2__) && !defined(__AVX512F__)
#include "avx2_computers.hpp"
#endif

#ifdef __AVX512F__
#include "avx512_computers.hpp"
#endif

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class DistanceComputer {};

template <>
class DistanceComputer<l2, f32> {
    using computer = SIMDComputer<l2, f32>;

  public:
    constexpr static auto VerticalPruning = computer::VerticalPruning<true>;
    constexpr static auto Vertical = computer::VerticalPruning<false>;
    constexpr static auto VerticalBlock = computer::Vertical;

    constexpr static auto VerticalBlockBatch4 = computer::VerticalBatch<4>;
    constexpr static auto VerticalBlockBatch8 = computer::VerticalBatch<8>;
    constexpr static auto VerticalBlockBatch16 = computer::VerticalBatch<16>;
    constexpr static auto VerticalBlockBatch32 = computer::VerticalBatch<32>;
    constexpr static auto VerticalBlockBatch64 = computer::VerticalBatch<64>;

    constexpr static auto VerticalBlockBatch8V2 = computer::VerticalBatchV2<8>;
    constexpr static auto VerticalBlockBatch64V2 = computer::VerticalBatchV2<64>;

    constexpr static auto VerticalBlockBatch64SIMD = computer::VerticalBatch64SIMD;

    constexpr static auto Horizontal = computer::Horizontal;
};

template <>
class DistanceComputer<l2, u8> {
    using computer = SIMDComputer<l2, u8>;

  public:
    constexpr static auto VerticalPruning = computer::VerticalPruning<true>;
    constexpr static auto Vertical = computer::VerticalPruning<false>;

    constexpr static auto VerticalBlock = computer::Vertical;
    constexpr static auto Horizontal = computer::Horizontal;
};

}; // namespace skmeans

#endif // SUPERKMEANS_BASE_COMPUTERS_HPP