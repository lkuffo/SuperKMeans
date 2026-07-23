// Force the GEMM rotation path (the reviewer's target): the orthogonal random
// rotation done through sgemm, which needs a separate output buffer.
#undef HAS_FFTW

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <string>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/pdx/adsampling.h"

namespace {

constexpr uint32_t DEFAULT_REPS = 3;
constexpr uint32_t SAMPLE_ROWS = 4096; // rows used for the correctness check

double AverageMilliseconds(size_t total_ns, uint32_t reps) {
    return static_cast<double>(total_ns) / reps / 1e6;
}

void PrintBytes(size_t bytes) {
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    if (mib >= 1024.0) {
        std::cout << std::fixed << std::setprecision(2) << (mib / 1024.0) << " GiB";
    } else {
        std::cout << std::fixed << std::setprecision(2) << mib << " MiB";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("mxbai");
    const uint32_t reps = (argc > 2) ? static_cast<uint32_t>(std::stoul(argv[2])) : DEFAULT_REPS;

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        return 1;
    }
    const size_t n = it->second.first;
    const uint32_t d = static_cast<uint32_t>(it->second.second);
    const std::string filename = bench_utils::get_data_path(dataset);

    const size_t threads = omp_get_max_threads();
    omp_set_num_threads(static_cast<int>(threads));
    skmeans::g_n_threads = static_cast<uint32_t>(threads);

    std::cout << "=== Orthogonal Random Rotation Microbenchmark (GEMM path) ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "Threads: " << threads << "  reps: " << reps
              << "  in-place block rows: " << skmeans::ROTATION_INPLACE_BLOCK_ROWS << "\n"
              << std::endl;

    auto data = std::unique_ptr<float[]>(new float[n * d]);
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.get()), static_cast<std::streamsize>(n * d) * sizeof(float));
    file.close();

    skmeans::ADSamplingPruner<skmeans::Quantization::f32> pruner(
        d, skmeans::PRUNER_INITIAL_THRESHOLD
    );

    // Scratch that also plays the role of the current approach's separate output buffer.
    auto work = std::unique_ptr<float[]>(new float[n * d]);

    // Ground-truth rotation of the first SAMPLE_ROWS via the trusted extra-buffer path.
    const uint32_t sample_rows = static_cast<uint32_t>(std::min<size_t>(SAMPLE_ROWS, n));
    auto reference = std::unique_ptr<float[]>(new float[static_cast<size_t>(sample_rows) * d]);
    pruner.Rotate(data.get(), reference.get(), sample_rows);

    // --- Current approach: extra full-size output buffer ---
    pruner.Rotate(data.get(), work.get(), static_cast<uint32_t>(n)); // warmup
    bench_utils::TicToc extra_buffer_timer;
    for (uint32_t r = 0; r < reps; ++r) {
        extra_buffer_timer.Tic();
        pruner.Rotate(data.get(), work.get(), static_cast<uint32_t>(n));
        extra_buffer_timer.Toc();
    }
    const double extra_buffer_ms = AverageMilliseconds(extra_buffer_timer.accum_time, reps);

    // --- In-place approach: streaming row-blocks (data copied into work, rotated in place) ---
    std::copy_n(data.get(), n * d, work.get());
    pruner.RotateInPlace(work.get(), static_cast<uint32_t>(n)); // warmup
    bench_utils::TicToc in_place_timer;
    for (uint32_t r = 0; r < reps; ++r) {
        std::copy_n(data.get(), n * d, work.get());
        in_place_timer.Tic();
        pruner.RotateInPlace(work.get(), static_cast<uint32_t>(n));
        in_place_timer.Toc();
    }
    const double in_place_ms = AverageMilliseconds(in_place_timer.accum_time, reps);

    // --- Correctness: last in-place result must match the extra-buffer reference ---
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(sample_rows) * d; ++i) {
        const float abs_diff = std::abs(work[i] - reference[i]);
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, abs_diff / std::max(std::abs(reference[i]), 1e-6f));
    }
    const bool correct = max_rel_diff < 1e-4f;

    const double gflop = 2.0 * static_cast<double>(n) * d * d / 1e9;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(24) << std::left << "Approach" << std::right << std::setw(14) << "Time (ms)"
              << std::setw(14) << "GFLOP/s" << std::setw(18) << "Extra memory" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::cout << std::setw(24) << std::left << "Extra buffer (current)" << std::right << std::setw(14)
              << extra_buffer_ms << std::setw(14) << (gflop / (extra_buffer_ms / 1e3)) << std::setw(13);
    PrintBytes(n * d * sizeof(float));
    std::cout << std::endl;

    std::cout << std::setw(24) << std::left << "In-place (streaming)" << std::right << std::setw(14)
              << in_place_ms << std::setw(14) << (gflop / (in_place_ms / 1e3)) << std::setw(13);
    PrintBytes(static_cast<size_t>(skmeans::ROTATION_INPLACE_BLOCK_ROWS) * d * sizeof(float));
    std::cout << std::endl;

    std::cout << std::string(70, '-') << std::endl;
    std::cout << "In-place / extra-buffer time ratio: " << (in_place_ms / extra_buffer_ms) << "x"
              << std::endl;
    std::cout << "Extra-memory reduction: "
              << (static_cast<double>(n) / skmeans::ROTATION_INPLACE_BLOCK_ROWS) << "x" << std::endl;
    std::cout << "\nCorrectness (" << sample_rows << " rows): " << (correct ? "PASSED" : "FAILED")
              << "  (max abs diff " << std::scientific << max_abs_diff << ", max rel diff "
              << max_rel_diff << ")" << std::endl;

    return correct ? 0 : 1;
}
