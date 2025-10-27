#ifndef SKMEANS_PDX_UTILS_HPP
#define SKMEANS_PDX_UTILS_HPP

#include <cassert>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <string>
#include <sys/stat.h>

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