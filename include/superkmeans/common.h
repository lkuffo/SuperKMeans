#pragma once
#ifndef SUPERKMEANS_COMMON_H
#define SUPERKMEANS_COMMON_H

#include <cinttypes>
#include <cstdint>
#include <cstdio>

#define SKMEANS_ENSURE_POSITIVE(x)                                                                 \
    if ((x) <= 0) {                                                                                \
        throw std::invalid_argument("Value must be positive: " #x);                                \
    }

#ifndef SKM_RESTRICT
#if defined(__GNUC__) || defined(__clang__)
#define SKM_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define SKM_RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
#define SKM_RESTRICT __restrict__
#else
#define SKM_RESTRICT
#endif
#endif

#ifndef SKM_ALWAYS_INLINE
#if __has_cpp_attribute(gnu::always_inline)
#define SKM_ALWAYS_INLINE [[gnu::always_inline]]
#elif defined(__GNUC__) || defined(__clang__)
#define SKM_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define SKM_ALWAYS_INLINE __forceinline
#else
#define SKM_ALWAYS_INLINE
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SKM_LIKELY(x) __builtin_expect(!!(x), 1)
#define SKM_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define SKM_LIKELY(x) (x)
#define SKM_UNLIKELY(x) (x)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SKM_PREFETCH(addr, rw, locality) __builtin_prefetch((addr), (rw), (locality))
#elif defined(_MSC_VER)
#include <xmmintrin.h>
#define SKM_PREFETCH(addr, rw, locality)                                                           \
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#else
#define SKM_PREFETCH(addr, rw, locality) ((void) 0)
#endif

namespace skmeans {

static inline constexpr float PROPORTION_VERTICAL_DIM = 0.75;
static inline constexpr size_t D_THRESHOLD_FOR_DCT_ROTATION = 512;
static inline constexpr size_t VECTOR_CHUNK_SIZE = 64;
static inline constexpr size_t H_DIM_SIZE = 64;
// Evaluating the pruning threshold is so fast that we can allow smaller fetching sizes
// to avoid more data access. Super useful in architectures with low bandwidth at L3/DRAM like
// Intel SPR
static constexpr uint32_t DIMENSIONS_FETCHING_SIZES[24] = {4,   4,   8,   8,   8,    16,
                                                           16,  32,  32,  32,  32,   64,
                                                           64,  64,  64,  128, 128,  128,
                                                           128, 256, 256, 512, 1024, 2048};

template <class T, T val = 8>
static constexpr uint32_t AlignValue(T n) {
    return ((n + (val - 1)) / val) * val;
}

enum DistanceFunction { l2, dp, neg_l2 };

enum Quantization { f32, u8, f16, bf16 };

template <Quantization q>
struct DistanceType {
    using type = uint32_t;
};
template <>
struct DistanceType<f32> {
    using type = float;
};
template <Quantization q>
using skmeans_distance_t = typename DistanceType<q>::type;

template <Quantization q>
struct DataType {
    using type = uint8_t;
};
template <>
struct DataType<f32> {
    using type = float;
};
template <Quantization q>
using skmeans_value_t = typename DataType<q>::type;

template <Quantization q>
struct CentroidDataType {
    using type = float;
};
template <>
struct CentroidDataType<f32> {
    using type = float;
};
template <Quantization q>
using skmeans_centroid_value_t = typename CentroidDataType<q>::type;

template <Quantization q>
struct KNNCandidate {
    uint32_t index;
    float distance;
};

template <Quantization q>
struct VectorComparator {
    bool operator()(const KNNCandidate<q>& a, const KNNCandidate<q>& b) {
        return a.distance < b.distance;
    }
};

template <Quantization q>
struct Cluster {
    uint32_t num_embeddings{};
    uint32_t* indices = nullptr;
    skmeans_value_t<u8>* data = nullptr;
};

template <>
struct Cluster<f32> {
    uint32_t num_embeddings{};
    uint32_t* indices = nullptr;
    skmeans_value_t<f32>* data = nullptr;
};

}; // namespace skmeans

#endif // SUPERKMEANS_COMMON_H
