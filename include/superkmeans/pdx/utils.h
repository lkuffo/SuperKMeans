#ifndef SKMEANS_PDX_UTILS_HPP
#define SKMEANS_PDX_UTILS_HPP

#include "superkmeans/common.h"
#include <Eigen/Eigen/Dense>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef linux
#include <linux/mman.h>
#endif

namespace skmeans {

template <Quantization q = f32, bool FULLY_TRANSPOSED = false>
inline void PDXify(
    const skmeans_value_t<q>* SKM_RESTRICT in_vectors, skmeans_value_t<q>* SKM_RESTRICT out_pdx_vectors,
    const size_t n, const size_t d
) {
    //std::cout << "Threads used in Eigen: " << Eigen::nbThreads() << "\n";
    assert(n % VECTOR_CHUNK_SIZE == 0);

    // We compute the split of vertical and horizontal dimensions
    size_t horizontal_d = static_cast<uint32_t>(d * PROPORTION_VERTICAL_DIM);
    size_t vertical_d = d - horizontal_d;
    if (horizontal_d % H_DIM_SIZE > 0) {
        horizontal_d = std::floor((1.0 * horizontal_d / H_DIM_SIZE) + 0.5) * H_DIM_SIZE;
        vertical_d = d - horizontal_d;
    }
    if (!vertical_d) {
        horizontal_d = H_DIM_SIZE;
        vertical_d = d - horizontal_d;
    }
    // std::cout << "Horizontal dimension: " << horizontal_d << "\n";
    // std::cout << "Vertical dimension: " << vertical_d << "\n";

    // TODO(@lkuffo, high): Parallelize
    // There are two options:
    // - Keep Eigen with 1 thread and use fork_union for paralellization
    // - Do everything here in just one Eigen operation on the big matrix (should use
    // paralellization with C++11 threads). BUT: We want to avoid OpenMP
    // Benchmark! --> I need to be at least AS FAST as numpy full transposition
    for (size_t i = 0; i < n; i += VECTOR_CHUNK_SIZE) {
        auto chunk_offset = i * d; // Chunk offset is the same in both layouts
        auto chunk_p = in_vectors + chunk_offset;
        auto out_chunk_p = out_pdx_vectors + chunk_offset;
        // Map vs Copying the buffer
        // Copying the buffer as a column major could be very beneficial for me
        auto _vectors = Eigen::Map<
            const Eigen::Matrix<skmeans_value_t<q>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            chunk_p, VECTOR_CHUNK_SIZE, d
        );
        // Vertical Part
        if constexpr (FULLY_TRANSPOSED) {
            auto _vertical_part = _vectors.transpose().eval();
            std::memcpy(out_chunk_p, _vertical_part.data(), VECTOR_CHUNK_SIZE * d);
            return;
        }
        auto _vertical_part = _vectors.leftCols(vertical_d).transpose().eval();
        std::memcpy(out_chunk_p, _vertical_part.data(), VECTOR_CHUNK_SIZE * vertical_d);
        out_chunk_p += VECTOR_CHUNK_SIZE * vertical_d;

        // Horizontal Parts
        for (size_t j = 0; j < horizontal_d; j += H_DIM_SIZE) {
            auto _horizontal_chunk = _vectors.block(0, j, VECTOR_CHUNK_SIZE, H_DIM_SIZE).eval();
            std::memcpy(out_chunk_p, _horizontal_chunk.data(), VECTOR_CHUNK_SIZE * H_DIM_SIZE);
            out_chunk_p += VECTOR_CHUNK_SIZE * H_DIM_SIZE;
        }
    }
}

/******************************************************************
 * File reader
 ******************************************************************/
inline std::unique_ptr<char[]> MmapFile(const std::string& filename) {
    struct stat file_stats{};
    int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd == -1)
        throw std::runtime_error("Failed to open file");

    fstat(fd, &file_stats);
    size_t file_size = file_stats.st_size;

    auto data = std::make_unique<char[]>(file_size);
    std::ifstream input(filename, std::ios::binary);
    input.read(data.get(), file_size);

    return data;
}

/******************************************************************
 * Clock to benchmark algorithms runtime
 ******************************************************************/
class TicToc {
  public:
    size_t accum_time = 0;
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();

    void Reset() {
        accum_time = 0;
        start = std::chrono::high_resolution_clock::now();
    }

    void Tic() { start = std::chrono::high_resolution_clock::now(); }

    void Toc() {
        auto end = std::chrono::high_resolution_clock::now();
        accum_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};
} // namespace skmeans

#endif // SKMEANS_PDX_UTILS_HPP