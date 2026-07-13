#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/distance_computers/batch_computers.h"
#include "superkmeans/pdx/utils.h"

// Number of points to sample for ID estimation
constexpr size_t N_SAMPLE = 50000;
// k values to evaluate for MLE and LID
constexpr int K_VALUES[] = {5, 10, 20, 50, 100};
constexpr size_t N_K_VALUES = sizeof(K_VALUES) / sizeof(K_VALUES[0]);
// Maximum k needed (for kNN precomputation).
// We request K_MAX+1 from BatchComputer to account for self-matches.
constexpr int K_MAX = 256;
constexpr int K_QUERY = K_MAX + 1;

using L2BatchComputer = skmeans::BatchComputer<skmeans::DistanceFunction::l2, skmeans::Quantization::f32>;

/**
 * @brief Two-NN estimator (Facco et al.), DADApy-equivalent.
 *
 * mu_i = r2/r1. Degenerate points (r1 = 0, i.e. duplicates) are dropped, and
 * the largest (1 - fraction) of mu values are trimmed before fitting.
 *   - id_ml:   MLE  d = N / sum(log mu)
 *   - id_base: least-squares slope of -log(1 - i/N) vs log(mu) through origin
 */
struct TwoNNEstimate {
    double id_ml_all;
    double id_ml_trim;
    double id_base_trim;
    size_t n_used;
    size_t n_dropped;
};

static TwoNNEstimate EstimateTwoNN(
    const std::vector<float>& knn_dists, size_t n, int k_max, double fraction
) {
    std::vector<double> log_mu;
    log_mu.reserve(n);
    size_t dropped = 0;
    for (size_t i = 0; i < n; ++i) {
        float r1 = knn_dists[i * k_max + 0];
        float r2 = knn_dists[i * k_max + 1];
        if (!(r1 > 0.0f) || !(r2 > 0.0f)) { ++dropped; continue; }
        double lm = std::log(static_cast<double>(r2) / static_cast<double>(r1));
        if (std::isfinite(lm) && lm >= 0.0) log_mu.push_back(lm);
        else ++dropped;
    }

    TwoNNEstimate est{0.0, 0.0, 0.0, 0, dropped};
    if (log_mu.size() < 2) return est;

    double sum_all = std::accumulate(log_mu.begin(), log_mu.end(), 0.0);
    if (sum_all > 0.0) est.id_ml_all = static_cast<double>(log_mu.size()) / sum_all;

    std::sort(log_mu.begin(), log_mu.end());
    const size_t N = log_mu.size();
    size_t n_eff = static_cast<size_t>(static_cast<double>(N) * fraction);
    n_eff = std::min(N, std::max<size_t>(2, n_eff));
    est.n_used = n_eff;

    double sum_trim = 0.0, sxy = 0.0, sxx = 0.0;
    for (size_t i = 0; i < n_eff; ++i) {
        double x = log_mu[i];
        double F = static_cast<double>(i + 1) / static_cast<double>(N);
        double y = -std::log(1.0 - F);
        sum_trim += x;
        sxy += x * y;
        sxx += x * x;
    }
    if (sum_trim > 0.0) est.id_ml_trim = static_cast<double>(n_eff) / sum_trim;
    if (sxx > 0.0) est.id_base_trim = sxy / sxx;
    return est;
}

/**
 * @brief Generalized Two-NN / Gride estimator (Denti et al. 2022).
 *
 * Uses the ratio of the (2k)-th to k-th NN distance, which is robust to the
 * neighborhood size (unlike the raw 1st/2nd-NN ratio):
 *   d_hat = (H_{2k-1} - H_{k-1}) / mean_i log(r_{i,2k} / r_{i,k})
 */
static double EstimateGride(const std::vector<float>& knn_dists, size_t n, int k_max, int k) {
    if (2 * k > k_max) return 0.0;
    double harmonic = 0.0;
    for (int j = k; j < 2 * k; ++j) harmonic += 1.0 / static_cast<double>(j);
    double sum_log = 0.0;
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
        float rk = knn_dists[i * k_max + (k - 1)];
        float r2k = knn_dists[i * k_max + (2 * k - 1)];
        if (!(rk > 0.0f) || !(r2k > 0.0f)) continue;
        double lr = std::log(static_cast<double>(r2k) / static_cast<double>(rk));
        if (std::isfinite(lr)) { sum_log += lr; ++valid; }
    }
    if (valid == 0 || sum_log <= 0.0) return 0.0;
    return harmonic / (sum_log / static_cast<double>(valid));
}

/**
 * @brief MLE estimator (Levina & Bickel 2004).
 *
 * For each point i with kNN distances r_{i,1} <= ... <= r_{i,k}:
 *   m_k(x_i) = [ 1/(k-1) * sum_{j=1}^{k-1} log(r_{i,k} / r_{i,j}) ]^{-1}
 *
 * Global estimate: ID = (1/n) * sum m_k(x_i)
 */
static double EstimateMLE(
    const std::vector<float>& knn_dists, size_t n, int k_max, int k
) {
    double sum_id = 0.0;
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
        float rk = knn_dists[i * k_max + (k - 1)];
        if (rk <= 0.0f) continue;
        double sum_log = 0.0;
        for (int j = 0; j < k - 1; ++j) {
            float rj = knn_dists[i * k_max + j];
            if (rj <= 0.0f) continue;
            sum_log += std::log(static_cast<double>(rk) / static_cast<double>(rj));
        }
        if (sum_log > 0.0) {
            sum_id += static_cast<double>(k - 1) / sum_log;
            ++valid;
        }
    }
    return (valid > 0) ? sum_id / static_cast<double>(valid) : 0.0;
}

/**
 * @brief LID MLE estimator (Amsaleg et al. 2015 / Houle 2017).
 *
 * For each point i with kNN distances r_{i,1} <= ... <= r_{i,k}:
 *   LID(x_i) = -k / sum_{j=1}^{k} log(r_{i,j} / r_{i,k})
 *
 * Global estimate: median of per-point LID values.
 */
static double EstimateLID(
    const std::vector<float>& knn_dists, size_t n, int k_max, int k
) {
    std::vector<double> lid_values;
    lid_values.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float rk = knn_dists[i * k_max + (k - 1)];
        if (rk <= 0.0f) continue;
        double sum_log = 0.0;
        int count = 0;
        for (int j = 0; j < k - 1; ++j) {
            float rj = knn_dists[i * k_max + j];
            if (rj <= 0.0f) continue;
            sum_log += std::log(static_cast<double>(rj) / static_cast<double>(rk));
            ++count;
        }
        // sum_log is negative (rj < rk), so -count/sum_log is positive
        if (sum_log < 0.0) {
            lid_values.push_back(-static_cast<double>(count) / sum_log);
        }
    }
    if (lid_values.empty()) return 0.0;

    std::sort(lid_values.begin(), lid_values.end());
    return lid_values[lid_values.size() / 2]; // median
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset> [n_sample]" << std::endl;
        return 1;
    }
    const std::string dataset = argv[1];

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset: " << dataset << std::endl;
        return 1;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")" << std::endl;

    // Load data
    std::string data_path = bench_utils::get_data_path(dataset);
    std::vector<float> data(n * d);
    {
        std::ifstream file(data_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open data file: " << data_path << std::endl;
            return 1;
        }
        file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    }

    // Sample points
    size_t n_sample_req = N_SAMPLE;
    if (argc >= 3) n_sample_req = static_cast<size_t>(std::stoull(argv[2]));
    const size_t n_sample = std::min(n_sample_req, n);
    std::vector<float> sampled(n_sample * d);
    {
        std::mt19937 rng(42);
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i = 0; i < n_sample; ++i) {
            std::memcpy(
                sampled.data() + i * d,
                data.data() + indices[i] * d,
                d * sizeof(float)
            );
        }
    }
    std::cout << "Sampled " << n_sample << " points for ID estimation" << std::endl;

    // Pre-compute squared L2 norms
    auto norms = skmeans::ComputeNorms(sampled.data(), n_sample, d);

    // Allocate scratch buffer for BatchComputer
    std::vector<float> tmp_buf(skmeans::X_BATCH_SIZE * skmeans::Y_BATCH_SIZE);

    // Compute (K_MAX+1)-NN via BLAS-accelerated BatchComputer (self-kNN includes self-match)
    std::cout << "Computing " << K_QUERY << "-NN distances (BLAS)..." << std::flush;
    bench_utils::TicToc timer;
    timer.Tic();

    std::vector<uint32_t> knn_indices(n_sample * K_QUERY);
    std::vector<float> knn_dists_raw(n_sample * K_QUERY);
    L2BatchComputer::FindKNearestNeighbors(
        sampled.data(), sampled.data(),
        n_sample, n_sample, d,
        norms.data(), norms.data(),
        K_QUERY,
        knn_indices.data(), knn_dists_raw.data(),
        tmp_buf.data()
    );

    // Post-process: strip self-matches and convert squared distances to L2 distances.
    // BatchComputer returns squared L2 distances. Each point's nearest neighbor list
    // may include itself (distance ~0). We skip self-matches and keep K_MAX neighbors.
    std::vector<float> knn_dists(n_sample * K_MAX);
    for (size_t i = 0; i < n_sample; ++i) {
        int out_idx = 0;
        for (int j = 0; j < K_QUERY && out_idx < K_MAX; ++j) {
            if (knn_indices[i * K_QUERY + j] == static_cast<uint32_t>(i)) continue;
            knn_dists[i * K_MAX + out_idx] = std::sqrt(knn_dists_raw[i * K_QUERY + j]);
            ++out_idx;
        }
    }

    timer.Toc();
    std::cout << " done (" << std::fixed << std::setprecision(1)
              << timer.GetMilliseconds() << " ms)" << std::endl;

    // ── Two-NN ──
    constexpr double TWO_NN_FRACTION = 0.9;
    TwoNNEstimate two_nn = EstimateTwoNN(knn_dists, n_sample, K_MAX, TWO_NN_FRACTION);
    std::cout << "\n=== Intrinsic Dimensionality Estimates ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nTwo-NN (Facco et al.), dropped " << two_nn.n_dropped
              << " degenerate pts, used " << two_nn.n_used << ":" << std::endl;
    std::cout << "  ML (all, no trim)      : " << two_nn.id_ml_all << std::endl;
    std::cout << "  ML (fraction=0.9)      : " << two_nn.id_ml_trim << std::endl;
    std::cout << "  base linear fit (0.9)  : " << two_nn.id_base_trim << std::endl;

    std::cout << "\nGride (generalized Two-NN, robust to neighborhood size):" << std::endl;
    for (int k : {2, 4, 8, 16, 32, 64, 128}) {
        std::cout << "  k=" << std::setw(3) << k << ", 2k=" << std::setw(3) << 2 * k
                  << " : " << EstimateGride(knn_dists, n_sample, K_MAX, k) << std::endl;
    }

    // ── MLE and LID for various k values ──
    std::cout << "\n" << std::setw(8) << "k"
              << std::setw(16) << "MLE (L&B)"
              << std::setw(16) << "LID (median)"
              << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    for (size_t ki = 0; ki < N_K_VALUES; ++ki) {
        int k = K_VALUES[ki];
        double mle = EstimateMLE(knn_dists, n_sample, K_MAX, k);
        double lid = EstimateLID(knn_dists, n_sample, K_MAX, k);
        std::cout << std::setw(8) << k
                  << std::setw(16) << mle
                  << std::setw(16) << lid
                  << std::endl;
    }

    // ── Suggested TARGET_D ──
    // Use MLE at k=20 as the primary estimate, round up to next multiple of 64
    double mle_k20 = EstimateMLE(knn_dists, n_sample, K_MAX, 20);
    size_t suggested = ((static_cast<size_t>(std::ceil(mle_k20)) + 63) / 64) * 64;
    suggested = std::min(suggested, d);
    std::cout << "\nSuggested TARGET_D (MLE@k=20 rounded to 64): " << suggested << std::endl;

    return 0;
}
