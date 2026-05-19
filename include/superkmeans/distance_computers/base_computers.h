#pragma once

#include "superkmeans/common.h"

#ifdef __ARM_NEON
#include "neon_computers.h"
#endif

#if defined(__AVX2__) && !defined(__AVX512F__)
#include "avx2_computers.h"
#endif

#ifdef __AVX512F__
#include "avx512_computers.h"
#endif

#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
#include "scalar_computers.h"
#endif

namespace skmeans {

template <DistanceFunction alpha, Quantization q>
class DistanceComputer {};

template <>
class DistanceComputer<DistanceFunction::l2, Quantization::f32> {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarComputer<DistanceFunction::l2, Quantization::f32>;
#else
    using computer = SIMDComputer<DistanceFunction::l2, Quantization::f32>;
#endif

  public:
    constexpr static auto Horizontal = computer::Horizontal;
};

template <>
class DistanceComputer<DistanceFunction::l2, Quantization::u8> {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarComputer<DistanceFunction::l2, Quantization::u8>;
#else
    using computer = SIMDComputer<DistanceFunction::l2, Quantization::u8>;
#endif

  public:
    constexpr static auto Horizontal = computer::Horizontal;
};

template <>
class DistanceComputer<DistanceFunction::l2, Quantization::u4> {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarComputer<DistanceFunction::l2, Quantization::u4>;
#else
    using computer = SIMDComputer<DistanceFunction::l2, Quantization::u4>;
#endif

  public:
    constexpr static auto Horizontal = computer::Horizontal;
};

template <>
class DistanceComputer<DistanceFunction::l2, Quantization::b8> {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarComputer<DistanceFunction::l2, Quantization::b8>;
#else
    using computer = SIMDComputer<DistanceFunction::l2, Quantization::b8>;
#endif

  public:
    constexpr static auto Horizontal = computer::Horizontal;
    constexpr static auto HorizontalMultiPlane = computer::HorizontalMultiPlane;
};

template <Quantization q>
class UtilsComputer {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarUtilsComputer<q>;
#else
    using computer = SIMDUtilsComputer<q>;
#endif

  public:
    constexpr static auto FlipSign = computer::FlipSign;
    constexpr static auto InitPositionsArray = computer::InitPositionsArray;
    constexpr static auto PackU8ToU4x2 = computer::PackU8ToU4x2;
    constexpr static auto UnpackU4x2ToU8 = computer::UnpackU4x2ToU8;
};


class FastScanComputer {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using computer = ScalarFastScanComputer;
#else
    using computer = SIMDFastScanComputer;
#endif

  public:
    static constexpr size_t kBlockSize = computer::kBlockSize;
    constexpr static auto ScanBlock = computer::ScanBlock<false>;
    constexpr static auto ScanBlockWide = computer::ScanBlock<true>;
    constexpr static auto RabitQCorrection = computer::RabitQCorrection<false>;
    constexpr static auto RabitQCorrectionU32 = computer::RabitQCorrection<true>;
    constexpr static auto RabitQCompactSurvivors = computer::RabitQCompactSurvivors;
    constexpr static auto RabitQCorrectionAndCompact = computer::RabitQCorrectionAndCompact<false>;
    constexpr static auto RabitQCorrectionAndCompactU32 = computer::RabitQCorrectionAndCompact<true>;

    template<int NBlocks>
    static void ScanBlockMulti(
        const uint8_t* const* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        uint16_t* const* out_dot
    ) {
        computer::template ScanBlockMulti<NBlocks>(packed, lut, binary_bytes, out_dot);
    }

    template<int NBlocks>
    static void ScanBlockMultiAndCorrectAndCompact(
        const uint8_t* const* packed,
        const uint8_t* lut,
        size_t binary_bytes,
        float c1j, float c34j, float qr_j, float neg2_c2j,
        const float* const* or_c_l2sqr,
        const float* const* neg2_dp,
        const float* const* dp_sum_q,
        const float* const* threshold,
        uint16_t* const* partial_dot_out,
        uint32_t* const* survivor_positions,
        size_t* n_survivors_out
    ) {
        computer::template ScanBlockMultiAndCorrectAndCompact<NBlocks>(
            packed, lut, binary_bytes,
            c1j, c34j, qr_j, neg2_c2j,
            or_c_l2sqr, neg2_dp, dp_sum_q, threshold,
            partial_dot_out, survivor_positions, n_survivors_out);
    }
};

class RaBitQCodec {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using codec = ScalarRaBitQCodec;
#else
    using codec = SIMDRaBitQCodec;
#endif

  public:
    constexpr static auto EncodeOne = codec::EncodeOne;
    constexpr static auto DecodeOne = codec::DecodeOne;
};

class LVQ4Codec {
#if !defined(__ARM_NEON) && !defined(__AVX2__) && !defined(__AVX512F__)
    using codec = ScalarLVQ4Codec;
#else
    using codec = SIMDLVQ4Codec;
#endif

  public:
    constexpr static auto EncodeOne = codec::EncodeOne;
    constexpr static auto DecodeOne = codec::DecodeOne;
};

} // namespace skmeans