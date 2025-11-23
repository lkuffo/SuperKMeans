#ifndef SKMEANS_PDX_UTILS_HPP
#define SKMEANS_PDX_UTILS_HPP

#include <cassert>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <ctime>

#ifdef linux
#include <linux/mman.h>
#endif

namespace skmeans {

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

inline uint32_t CeilXToMultipleOfM(uint32_t x, uint32_t m) {
    return (m == 0) ? x : ((x + m - 1) / m) * m;
}

inline uint32_t FloorXToMultipleOfM(uint32_t x, uint32_t m) {
    return (m == 0) ? x : (x / m) * m;
}

struct TicToc {
    uint64_t accum_time = 0;
    struct timespec start{};

    void Tic() {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    }

    void Toc() {
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        uint64_t d = (uint64_t)(end.tv_sec - start.tv_sec) * 1000000000ull
                   + (uint64_t)(end.tv_nsec - start.tv_nsec);
        accum_time += d;
    }

    void Reset() { accum_time = 0; }
};

/******************************************************************
 * Clock to benchmark algorithms runtime
 ******************************************************************/
class TicToc2 {
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