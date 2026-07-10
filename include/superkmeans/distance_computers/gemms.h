#pragma once

#include "superkmeans/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <vector>

#include <numkong/numkong.h>
#include "ruy/ruy.h"

namespace skmeans {

/**
 * @brief u8×u8→u32 dot-product GEMM leaf, dispatching between NumKong and ruy.
 *
 * Shared by SQ8 (native u8 codes) and LVQ4 (u4 codes decoded to u8). The caller
 * decides the backend via `use_numkong`; the NumKong path packs `b` into
 * `packed_buf` (skipped when `pack_b` is false, i.e. b is unchanged across
 * calls) and parallelizes the row dots. OMP parallelization is handled here.
 */
inline void U8Gemm(
    const uint8_t* a,
    const uint8_t* b,
    uint32_t* out,
    size_t m,
    size_t n,
    size_t k,
    size_t a_stride,
    size_t b_stride,
    bool use_numkong,
    std::vector<char>& packed_buf,
    bool pack_b
) {
    if (use_numkong) {
        if (pack_b) {
            const size_t pack_size = nk_dots_packed_size_u8(n, k);
            if (pack_size > packed_buf.size()) packed_buf.resize(pack_size);
            nk_dots_pack_u8(b, n, k, b_stride, packed_buf.data());
        }

        const size_t c_stride = n * sizeof(uint32_t);

#pragma omp parallel num_threads(g_n_threads)
        {
            nk_configure_thread(nk_capabilities());
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            size_t rows_per_t = (m + nt - 1) / nt;
            size_t start = tid * rows_per_t;
            size_t count = std::min(rows_per_t, m - start);
            if (start < m && count > 0) {
                nk_dots_packed_u8(
                    a + start * a_stride,
                    packed_buf.data(),
                    out + start * n,
                    count, n, k,
                    a_stride, c_stride
                );
            }
        }
        return;
    }

#pragma omp parallel for num_threads(g_n_threads) schedule(static)
    for (int t = 0; t < static_cast<int>(g_n_threads); ++t) {
        const size_t row_start = t * m / g_n_threads;
        const size_t row_end = (t + 1) * m / g_n_threads;
        const size_t local_rows = row_end - row_start;
        if (local_rows == 0) continue;

        thread_local ruy::Context ctx;
        ctx.set_max_num_threads(1);

        ruy::Matrix<std::uint8_t> lhs;
        lhs.mutable_layout()->set_rows(local_rows);
        lhs.mutable_layout()->set_cols(k);
        lhs.mutable_layout()->set_order(ruy::Order::kRowMajor);
        lhs.mutable_layout()->set_stride(a_stride);
        lhs.set_data(a + row_start * a_stride);

        ruy::Matrix<std::uint8_t> rhs;
        rhs.mutable_layout()->set_rows(k);
        rhs.mutable_layout()->set_cols(n);
        rhs.mutable_layout()->set_order(ruy::Order::kColMajor);
        rhs.mutable_layout()->set_stride(b_stride);
        rhs.set_data(b);

        ruy::Matrix<std::int32_t> dst;
        dst.mutable_layout()->set_rows(local_rows);
        dst.mutable_layout()->set_cols(n);
        dst.mutable_layout()->set_order(ruy::Order::kRowMajor);
        dst.mutable_layout()->set_stride(n);
        dst.set_data(reinterpret_cast<std::int32_t*>(out + row_start * n));

        ruy::MulParams<std::int32_t, std::int32_t> mul_params;
        ruy::Mul(lhs, rhs, mul_params, &ctx, &dst);
    }
}

/**
 * @brief u4×u4→u32 dot-product GEMM leaf via NumKong (no ruy fallback).
 *
 * Used by LVQ4 on x86 without AMX for wide matrices, where the packed u4 codes
 * are fed directly to NumKong. `a`/`b` point to packed u4x2 bytes; the NumKong
 * path packs `b` into `packed_buf` (skipped when `pack_b` is false) and
 * parallelizes the row dots. Strides are in nk_u4x2_t units.
 */
inline void U4Gemm(
    const uint8_t* a,
    const uint8_t* b,
    uint32_t* out,
    size_t m,
    size_t n,
    size_t k,
    size_t a_stride,
    size_t b_stride,
    std::vector<char>& packed_buf,
    bool pack_b
) {
    const auto* a_u4 = reinterpret_cast<const nk_u4x2_t*>(a);
    const auto* b_u4 = reinterpret_cast<const nk_u4x2_t*>(b);

    if (pack_b) {
        const size_t pack_size = nk_dots_packed_size_u4(n, k);
        if (pack_size > packed_buf.size()) packed_buf.resize(pack_size);
        nk_dots_pack_u4(b_u4, n, k, b_stride, packed_buf.data());
    }

    const size_t c_stride = n * sizeof(uint32_t);

#pragma omp parallel num_threads(g_n_threads)
    {
        nk_configure_thread(nk_capabilities());
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        size_t rows_per_t = (m + nt - 1) / nt;
        size_t start = tid * rows_per_t;
        size_t count = std::min(rows_per_t, m - start);
        if (start < m && count > 0) {
            nk_dots_packed_u4(
                a_u4 + start * a_stride,
                packed_buf.data(),
                out + start * n,
                count, n, k,
                a_stride, c_stride
            );
        }
    }
}

} // namespace skmeans
