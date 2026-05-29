// Empirical analysis of RabitQ centroid behavior with vs. without
// full_precision_final_centroids. Two competing hypotheses:
//
//   A. Coordinate-frame mismatch: centroids returned in (x - mean) space.
//      → delta[k] = c_fp[k] - c_q[k] is approximately the global mean for
//        every k (a constant per-coordinate offset).
//
//   B. Position shrinkage: centroids returned in raw space but pulled toward
//      the global mean by quantized-domain accumulation.
//      → ||c_q - mean|| is systematically smaller than ||c_fp - mean||;
//        delta[k] varies across centroids and points outward.
//
// We also check whether small ||c - mean|| correlates with large cluster size
// under FP=false (Check 3 — RabitQ codes near the mean are sign-bit ambiguous,
// so those centroids become assignment magnets).

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME = true
#endif

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <random>
#include <vector>

#include "bench_utils.h"
#include "superkmeans/common.h"
#include "superkmeans/superkmeans.h"

namespace {

std::vector<float> ComputeGlobalMean(const float* data, size_t n, size_t d) {
    std::vector<double> acc(d, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) acc[j] += data[i * d + j];
    }
    std::vector<float> mean(d);
    for (size_t j = 0; j < d; ++j) mean[j] = static_cast<float>(acc[j] / static_cast<double>(n));
    return mean;
}

std::vector<double> NormFromMean(
    const float* centroids, const std::vector<float>& mean, size_t k, size_t d
) {
    std::vector<double> norms(k, 0.0);
    for (size_t c = 0; c < k; ++c) {
        double s = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const double diff = centroids[c * d + j] - mean[j];
            s += diff * diff;
        }
        norms[c] = std::sqrt(s);
    }
    return norms;
}

std::vector<size_t> ClusterSizes(const std::vector<uint32_t>& assignments, size_t k) {
    std::vector<size_t> sizes(k, 0);
    for (uint32_t a : assignments) if (a < k) ++sizes[a];
    return sizes;
}

double Pearson(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t n = x.size();
    if (n == 0 || y.size() != n) return 0.0;
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
    mx /= static_cast<double>(n); my /= static_cast<double>(n);
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dx = x[i] - mx; const double dy = y[i] - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    if (sxx <= 0.0 || syy <= 0.0) return 0.0;
    return sxy / std::sqrt(sxx * syy);
}

double Spearman(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0;
    auto rank = [](const std::vector<double>& v) {
        std::vector<size_t> idx(v.size()); std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
        std::vector<double> r(v.size());
        for (size_t i = 0; i < idx.size(); ++i) r[idx[i]] = static_cast<double>(i);
        return r;
    };
    return Pearson(rank(x), rank(y));
}

double Quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    const double pos = q * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    std::nth_element(v.begin(), v.begin() + static_cast<long>(lo), v.end());
    const double v_lo = v[lo];
    if (lo == hi) return v_lo;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(hi), v.end());
    const double v_hi = v[hi];
    return v_lo + (pos - static_cast<double>(lo)) * (v_hi - v_lo);
}

void PrintDistribution(const std::string& label, std::vector<double> v) {
    if (v.empty()) { std::cout << label << " (empty)\n"; return; }
    double mn = v[0], mx = v[0], sum = 0.0;
    for (double x : v) { mn = std::min(mn, x); mx = std::max(mx, x); sum += x; }
    const double avg = sum / static_cast<double>(v.size());
    double sq = 0.0;
    for (double x : v) { const double dd = x - avg; sq += dd * dd; }
    const double sd = std::sqrt(sq / static_cast<double>(v.size()));
    std::cout << "  " << std::setw(34) << std::left << label << std::right
              << " mean=" << std::fixed << std::setprecision(4) << std::setw(9) << avg
              << " std=" << std::setw(9) << sd
              << " min=" << std::setw(9) << mn
              << " p50=" << std::setw(9) << Quantile(v, 0.50)
              << " max=" << std::setw(9) << mx << "\n";
}

struct TrainResult {
    std::vector<float> centroids;
    std::vector<uint32_t> q_assignments;
    double wcss = 0.0;
};

TrainResult RunTraining(
    const float* data, size_t n, size_t d, size_t n_clusters, int n_iters,
    bool blas_only, bool angular, bool full_precision_final_centroids, uint32_t seed
) {
    skmeans::SuperKMeansConfig config;
    config.iters = n_iters;
    config.verbose = false;
    config.verbose_detail = false;
    config.n_threads = omp_get_max_threads();
    config.unrotate_centroids = true;
    config.early_termination = false;
    config.sampling_fraction = 1.0f;
    config.tol = 1e-3f;
    config.quantizer_type = skmeans::QuantizerType::rabitq;
    config.quantized_centroid_update = true;
    config.full_precision_final_centroids = full_precision_final_centroids;
    config.use_blas_only = blas_only;
    config.angular = angular;
    config.seed = seed;
    // Skip the internal ADSampling rotation so that raw-space analysis matches
    // the algorithm's working space (rotation would change L1 norms used in dp_multiplier).
    config.data_already_rotated = true;

    auto kmeans =
        skmeans::SuperKMeans<skmeans::Quantization::u8, skmeans::DistanceFunction::l2>(
            n_clusters, d, config);
    TrainResult r;
    r.centroids = kmeans.Train(const_cast<float*>(data), n);
    r.q_assignments = kmeans.QuantizedAssign(
        const_cast<float*>(data), r.centroids.data(), n, n_clusters);
    using SKM = skmeans::SuperKMeans<skmeans::Quantization::u8, skmeans::DistanceFunction::l2>;
    r.wcss = SKM::ComputeWCSS(data, r.centroids.data(), r.q_assignments.data(), n, d);
    return r;
}

} // namespace

int main(int argc, char* argv[]) {
    const std::string algorithm = "rabitq_centroid_analysis";
    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("mxbai");
    bool blas_only = !(argc > 2 && std::string(argv[2]) == "pruning");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n"; return 1;
    }
    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters = bench_utils::get_default_n_clusters(n);
    const int n_iters = 10;
    const uint32_t seed = 42;
    std::string filename = bench_utils::get_data_path(dataset);
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);

    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(), bench_utils::ANGULAR_DATASETS.end(), dataset);
    const bool angular = (is_angular != bench_utils::ANGULAR_DATASETS.end());

    std::cout << "=== " << algorithm << " ===\n"
              << "Dataset: " << dataset << " n=" << n << " d=" << d
              << " k=" << n_clusters << " iters=" << n_iters
              << (blas_only ? " [BLAS-only]" : " [pruning]")
              << (angular ? " [angular]" : "") << "\n\n";

    std::vector<float> data;
    try { data.reserve(n * d); } catch (const std::bad_alloc& e) {
        std::cerr << "alloc failed: " << e.what() << "\n"; return 1;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "open failed: " << filename << "\n"; return 1; }
    file.read(reinterpret_cast<char*>(data.data()), n * d * sizeof(float));
    file.close();

    std::cout << "[1/2] Training with full_precision_final_centroids = false ...\n";
    bench_utils::TicToc t1; t1.Tic();
    auto run_q = RunTraining(data.data(), n, d, n_clusters, n_iters, blas_only, angular,
                             /*FP=*/false, seed);
    t1.Toc();
    std::cout << "      done in " << t1.GetMilliseconds() << " ms, WCSS=" << run_q.wcss << "\n";

    std::cout << "[2/2] Training with full_precision_final_centroids = true  ...\n";
    bench_utils::TicToc t2; t2.Tic();
    auto run_fp = RunTraining(data.data(), n, d, n_clusters, n_iters, blas_only, angular,
                              /*FP=*/true, seed);
    t2.Toc();
    std::cout << "      done in " << t2.GetMilliseconds() << " ms, WCSS=" << run_fp.wcss << "\n\n";

    // Global mean (the RabitQ centering vector).
    auto global_mean = ComputeGlobalMean(data.data(), n, d);
    double mean_l2 = 0.0;
    for (float v : global_mean) mean_l2 += static_cast<double>(v) * v;
    mean_l2 = std::sqrt(mean_l2);

    // Per-coordinate average of centroids (across the k centroids). If centroids
    // live in (x - mean) space, this average is ≈ 0; if in raw space, ≈ mean.
    std::vector<double> avg_centroid_q(d, 0.0), avg_centroid_fp(d, 0.0);
    for (size_t c = 0; c < n_clusters; ++c) {
        for (size_t j = 0; j < d; ++j) {
            avg_centroid_q[j]  += run_q.centroids[c * d + j];
            avg_centroid_fp[j] += run_fp.centroids[c * d + j];
        }
    }
    for (size_t j = 0; j < d; ++j) {
        avg_centroid_q[j]  /= static_cast<double>(n_clusters);
        avg_centroid_fp[j] /= static_cast<double>(n_clusters);
    }
    auto l2 = [&](const std::vector<double>& v) {
        double s = 0.0; for (double x : v) s += x * x; return std::sqrt(s);
    };
    auto l2_diff_to_mean = [&](const std::vector<double>& v) {
        double s = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const double dd = v[j] - static_cast<double>(global_mean[j]);
            s += dd * dd;
        }
        return std::sqrt(s);
    };

    std::cout << "════════════════════════════════════════════════════════════════\n"
              << "Hypothesis A — coordinate-frame check\n"
              << "════════════════════════════════════════════════════════════════\n"
              << "  ||global_mean||                       = " << std::fixed << std::setprecision(4)
              << mean_l2 << "\n"
              << "  ||avg_centroid_q  - 0||               = " << l2(avg_centroid_q)  << "\n"
              << "  ||avg_centroid_q  - global_mean||     = " << l2_diff_to_mean(avg_centroid_q)  << "\n"
              << "  ||avg_centroid_fp - 0||               = " << l2(avg_centroid_fp) << "\n"
              << "  ||avg_centroid_fp - global_mean||     = " << l2_diff_to_mean(avg_centroid_fp) << "\n"
              << "  (if centroids live in (x-mean) space, avg should be near 0;\n"
              << "   if in raw space, avg should be near global_mean)\n\n";

    // delta[k] = c_fp[k] - c_q[k]. If A holds, this vector is ≈ global_mean
    // for every k (and ≈ identical across k).
    std::vector<std::vector<double>> delta(n_clusters, std::vector<double>(d, 0.0));
    std::vector<double> delta_norm(n_clusters), delta_dot_mean(n_clusters);
    std::vector<double> delta_minus_mean_norm(n_clusters);
    for (size_t c = 0; c < n_clusters; ++c) {
        double dn = 0.0, dot = 0.0, dmm = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const double dlt = static_cast<double>(run_fp.centroids[c * d + j])
                             - static_cast<double>(run_q.centroids[c * d + j]);
            delta[c][j] = dlt;
            dn += dlt * dlt;
            dot += dlt * static_cast<double>(global_mean[j]);
            const double rd = dlt - static_cast<double>(global_mean[j]);
            dmm += rd * rd;
        }
        delta_norm[c] = std::sqrt(dn);
        delta_dot_mean[c] = dot;
        delta_minus_mean_norm[c] = std::sqrt(dmm);
    }
    std::cout << "  delta[k] := c_fp[k] - c_q[k]\n";
    PrintDistribution("||delta[k]||",                delta_norm);
    PrintDistribution("||delta[k] - global_mean||",  delta_minus_mean_norm);
    std::cout << "  If hypothesis A holds: ||delta - global_mean|| << ||delta|| (delta ≈ mean for all k).\n"
              << "  If hypothesis B holds: ||delta|| varies across k and is NOT close to mean.\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // Hypothesis B — shrinkage of ||c - mean||.
    // ════════════════════════════════════════════════════════════════════════
    auto norm_q  = NormFromMean(run_q.centroids.data(),  global_mean, n_clusters, d);
    auto norm_fp = NormFromMean(run_fp.centroids.data(), global_mean, n_clusters, d);

    std::cout << "════════════════════════════════════════════════════════════════\n"
              << "Hypothesis B — magnitude (||c - mean||) comparison\n"
              << "════════════════════════════════════════════════════════════════\n";
    PrintDistribution("||c - mean||  (FP=false)", norm_q);
    PrintDistribution("||c - mean||  (FP=true) ", norm_fp);

    std::vector<double> ratio(n_clusters);
    size_t shrunk = 0;
    for (size_t c = 0; c < n_clusters; ++c) {
        ratio[c] = (norm_fp[c] > 0.0) ? norm_q[c] / norm_fp[c] : 1.0;
        if (ratio[c] < 1.0) ++shrunk;
    }
    PrintDistribution("ratio  ||c_q-m|| / ||c_fp-m||", ratio);
    std::cout << "  centroids shrunk (ratio < 1): " << shrunk << " / " << n_clusters
              << "  (" << std::fixed << std::setprecision(1)
              << (100.0 * shrunk / n_clusters) << "%)\n";
    std::cout << "  Spearman(norm_fp,  ratio)  = " << std::setprecision(4)
              << Spearman(norm_fp, ratio) << "\n"
              << "  (negative ⇒ farther-out centroids shrink more, supporting B)\n\n";

    // ════════════════════════════════════════════════════════════════════════
    // Check 3 — cluster size vs centroid magnitude.
    // ════════════════════════════════════════════════════════════════════════
    auto sizes_q  = ClusterSizes(run_q.q_assignments,  n_clusters);
    auto sizes_fp = ClusterSizes(run_fp.q_assignments, n_clusters);
    std::vector<double> sizes_q_d(n_clusters), sizes_fp_d(n_clusters);
    for (size_t c = 0; c < n_clusters; ++c) {
        sizes_q_d[c]  = static_cast<double>(sizes_q[c]);
        sizes_fp_d[c] = static_cast<double>(sizes_fp[c]);
    }

    std::cout << "════════════════════════════════════════════════════════════════\n"
              << "Check 3 — cluster size vs centroid magnitude\n"
              << "════════════════════════════════════════════════════════════════\n"
              << std::fixed << std::setprecision(4)
              << "  FP=false:  pearson(||c-m||, size) = " << Pearson(norm_q,  sizes_q_d)  << "\n"
              << "             spearman(||c-m||,size) = " << Spearman(norm_q, sizes_q_d)  << "\n"
              << "  FP=true :  pearson(||c-m||, size) = " << Pearson(norm_fp, sizes_fp_d) << "\n"
              << "             spearman(||c-m||,size) = " << Spearman(norm_fp, sizes_fp_d) << "\n"
              << "  (negative correlation ⇒ small ||c-m|| ⇒ big cluster, supports magnet effect)\n\n";

    auto stratify = [&](const std::vector<double>& norms,
                        const std::vector<size_t>& sizes,
                        const std::string& label) {
        std::vector<size_t> idx(n_clusters);
        std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return norms[a] < norms[b]; });
        constexpr size_t BINS = 5;
        const size_t per_bin = (n_clusters + BINS - 1) / BINS;
        std::cout << "  " << label << " — centroids sorted by ||c-mean||, " << BINS << " quintiles:\n"
                  << "     bin |  ||c-mean|| range      |  mean size  |  median  |   max\n";
        for (size_t b = 0; b < BINS; ++b) {
            const size_t lo = b * per_bin;
            const size_t hi = std::min(n_clusters, lo + per_bin);
            if (lo >= hi) break;
            const double nmin = norms[idx[lo]];
            const double nmax = norms[idx[hi - 1]];
            std::vector<double> ssub; ssub.reserve(hi - lo);
            double s_sum = 0.0; size_t s_max = 0;
            for (size_t i = lo; i < hi; ++i) {
                const size_t s = sizes[idx[i]];
                ssub.push_back(static_cast<double>(s));
                s_sum += static_cast<double>(s);
                s_max = std::max(s_max, s);
            }
            const double s_mean = s_sum / static_cast<double>(ssub.size());
            const double s_med  = Quantile(ssub, 0.5);
            std::cout << "     Q" << (b + 1) << "  | "
                      << std::setw(8) << std::fixed << std::setprecision(2) << nmin
                      << " -- " << std::setw(8) << nmax
                      << "  | " << std::setw(9) << std::setprecision(1) << s_mean
                      << "  | " << std::setw(7) << std::setprecision(0) << s_med
                      << "  | " << std::setw(6) << s_max << "\n";
        }
    };
    stratify(norm_q,  sizes_q,  "FP=false");
    stratify(norm_fp, sizes_fp, "FP=true ");

    // ════════════════════════════════════════════════════════════════════════
    // Direct test of the sign-alignment mechanism.
    // For each cluster k (using FP=true assignments → "true" membership):
    //   sign_alignment_k = ||avg sign(x_i - m)||_2 / sqrt(d)   ∈ [0, 1]
    // 1.0 ⇒ all sign vectors in the cluster are identical (perfect lock).
    // 0.0 ⇒ sign vectors are random → averaging cancels them out.
    // If the mechanism is correct, this should correlate strongly with both
    // (a) ||c - m||  (outer clusters → more locked)
    // (b) per-cluster observed inflation (more locked → more residual inflation)
    // ════════════════════════════════════════════════════════════════════════
    std::cout << "════════════════════════════════════════════════════════════════\n"
              << "Sign-alignment mechanism — direct test\n"
              << "════════════════════════════════════════════════════════════════\n";
    {
        std::vector<double> sign_sum(static_cast<size_t>(n_clusters) * d, 0.0);
        std::vector<size_t> ck_count(n_clusters, 0);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t k = run_fp.q_assignments[i];
            if (k >= n_clusters) continue;
            ++ck_count[k];
            double* row = sign_sum.data() + static_cast<size_t>(k) * d;
            for (size_t j = 0; j < d; ++j) {
                const float diff = data[i * d + j] - global_mean[j];
                row[j] += (diff > 0.0f) ? 1.0 : -1.0;
            }
        }
        std::vector<double> sign_alignment(n_clusters, 0.0);
        for (size_t k = 0; k < n_clusters; ++k) {
            if (ck_count[k] == 0) continue;
            const double inv = 1.0 / static_cast<double>(ck_count[k]);
            double l2sq = 0.0;
            const double* row = sign_sum.data() + static_cast<size_t>(k) * d;
            for (size_t j = 0; j < d; ++j) {
                const double v = row[j] * inv;
                l2sq += v * v;
            }
            sign_alignment[k] = std::sqrt(l2sq / static_cast<double>(d));
        }
        PrintDistribution("sign_alignment_k (in [0,1])    ", sign_alignment);
        std::cout << "  Spearman(||c_fp-m||,  alignment)  = "
                  << std::fixed << std::setprecision(4)
                  << Spearman(norm_fp, sign_alignment) << "\n"
                  << "  Pearson (||c_fp-m||,  alignment)  = "
                  << Pearson(norm_fp, sign_alignment) << "\n"
                  << "  Spearman(alignment, obs_inflation) = "
                  << Spearman(sign_alignment, ratio) << "\n"
                  << "  Pearson (alignment, obs_inflation) = "
                  << Pearson(sign_alignment, ratio) << "\n";

        // Quintile view by alignment.
        std::vector<size_t> idx(n_clusters);
        std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return sign_alignment[a] < sign_alignment[b]; });
        constexpr size_t BINS = 5;
        const size_t per_bin = (n_clusters + BINS - 1) / BINS;
        std::cout << "  centroids sorted by alignment, " << BINS << " quintiles:\n"
                  << "     bin |  alignment range     |  mean ||c-m||  |  mean inflation\n";
        for (size_t b = 0; b < BINS; ++b) {
            const size_t lo = b * per_bin;
            const size_t hi = std::min(static_cast<size_t>(n_clusters), lo + per_bin);
            if (lo >= hi) break;
            double a_lo = sign_alignment[idx[lo]], a_hi = sign_alignment[idx[hi - 1]];
            double sum_norm = 0.0, sum_infl = 0.0;
            for (size_t i = lo; i < hi; ++i) {
                sum_norm += norm_fp[idx[i]];
                sum_infl += ratio[idx[i]];
            }
            const size_t n_bin = hi - lo;
            std::cout << "     Q" << (b + 1) << "  | "
                      << std::setw(8) << std::fixed << std::setprecision(4) << a_lo
                      << " - " << std::setw(8) << a_hi
                      << "  | " << std::setw(10) << std::setprecision(3)
                      << sum_norm / static_cast<double>(n_bin)
                      << "  | " << std::setw(10) << std::setprecision(4)
                      << sum_infl / static_cast<double>(n_bin) << "\n";
        }
    }
    std::cout << "\n";

    // ════════════════════════════════════════════════════════════════════════
    // Proof: analytical prediction of the FP=false centroid from RabitQ
    // encode + decode + average. If the prediction matches the observed
    // FP=false centroid, the mechanism is fully characterized.
    // ════════════════════════════════════════════════════════════════════════
    std::cout << "════════════════════════════════════════════════════════════════\n"
              << "Proof — analytical prediction of c_q from RabitQ encode/decode\n"
              << "════════════════════════════════════════════════════════════════\n";

    const double sqrt_d = std::sqrt(static_cast<double>(d));

    // Per-vector dp_multiplier_i and per-vector inflation ratio.
    // dp_i = ||x_i - m||² * sqrt(d) / ||x_i - m||_1
    // inflation_i = dp_i / ||x_i - m||  (Cauchy–Schwarz: >= 1, equality iff |x-m|_j is constant in j)
    std::vector<double> per_vec_inflation(n);
    std::vector<double> dp_all(n);
    std::vector<double> scale2_all(n);
    std::vector<double> l1_all(n);
    {
        double acc_inflation = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double scale2 = 0.0, l1 = 0.0;
            for (size_t j = 0; j < d; ++j) {
                const double diff = static_cast<double>(data[i * d + j])
                                  - static_cast<double>(global_mean[j]);
                scale2 += diff * diff;
                l1 += std::abs(diff);
            }
            scale2_all[i] = scale2;
            l1_all[i] = l1;
            const double scale = std::sqrt(scale2);
            const double dp = (l1 > 0.0) ? (scale2 * sqrt_d / l1) : 0.0;
            dp_all[i] = dp;
            per_vec_inflation[i] = (scale > 0.0) ? dp / scale : 1.0;
            acc_inflation += per_vec_inflation[i];
        }
        (void)acc_inflation;
    }
    PrintDistribution("per-vector dp / ||x-m||  (>= 1)", per_vec_inflation);

    // For each cluster k under FP=false assignments, accumulate:
    //   c_true_k = (1/|k|) Σ x_i           (raw mean — should match FP=true centroid if assignments align)
    //   c_pred_k = (1/|k|) Σ decoded_i     (RabitQ averaged decode — should match FP=false centroid)
    // Decode formula (matches ScalarRaBitQCodec::DecodeOne):
    //   decoded_i[j] = m[j] + (dp_i / sqrt(d)) * (±1 from sign(x_i_j - m_j))
    std::vector<float> c_true_local(n_clusters * d, 0.0f);
    std::vector<float> c_pred_local(n_clusters * d, 0.0f);
    std::vector<size_t> cs_under_q(n_clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t k = run_q.q_assignments[i];
        if (k >= n_clusters) continue;
        ++cs_under_q[k];
        const double inv_sqrt_d_dp = dp_all[i] / sqrt_d; // scale factor for sign vector
        float* row_true = c_true_local.data() + static_cast<size_t>(k) * d;
        float* row_pred = c_pred_local.data() + static_cast<size_t>(k) * d;
        for (size_t j = 0; j < d; ++j) {
            const float xv = data[i * d + j];
            row_true[j] += xv;
            const float diff = xv - global_mean[j];
            const float sg = (diff > 0.0f) ? 1.0f : -1.0f;
            // decoded_i[j] - m[j] = (dp_i / sqrt(d)) * sign(x_i_j - m_j)
            // Accumulate decoded_i[j] directly (we'll subtract m later when computing norms)
            row_pred[j] += static_cast<float>(global_mean[j] + inv_sqrt_d_dp * sg);
        }
    }
    for (size_t k = 0; k < n_clusters; ++k) {
        if (cs_under_q[k] == 0) continue;
        const float inv = 1.0f / static_cast<float>(cs_under_q[k]);
        float* row_true = c_true_local.data() + k * d;
        float* row_pred = c_pred_local.data() + k * d;
        for (size_t j = 0; j < d; ++j) {
            row_true[j] *= inv;
            row_pred[j] *= inv;
        }
    }

    // Norms ‖· − m‖ for predicted and observed.
    auto norm_pred = NormFromMean(c_pred_local.data(), global_mean, n_clusters, d);
    auto norm_true_local = NormFromMean(c_true_local.data(), global_mean, n_clusters, d);

    // Per-cluster errors between predicted-c_pred-and-observed-c_q.
    std::vector<double> pred_obs_abs_err(n_clusters);  // |‖pred-m‖ − ‖obs_q-m‖|
    std::vector<double> pred_obs_rel_err(n_clusters);
    for (size_t k = 0; k < n_clusters; ++k) {
        const double err = std::abs(norm_pred[k] - norm_q[k]);
        pred_obs_abs_err[k] = err;
        pred_obs_rel_err[k] = (norm_q[k] > 0.0) ? err / norm_q[k] : 0.0;
    }
    PrintDistribution("|‖pred-m‖ - ‖obs_q-m‖|         ", pred_obs_abs_err);
    PrintDistribution("rel err  ↑ / ‖obs_q-m‖         ", pred_obs_rel_err);
    std::cout << "  pearson (norm_pred, norm_obs_q)   = " << std::setprecision(4)
              << Pearson(norm_pred, norm_q) << "\n"
              << "  spearman(norm_pred, norm_obs_q)   = "
              << Spearman(norm_pred, norm_q) << "\n";

    // Sanity check: FP=true centroids should equal raw-mean per FP=true assignments.
    // We computed c_true_local using FP=false assignments; a more honest sanity is
    // to use FP=true assignments. We compute and report both.
    std::vector<float> c_true_fp(n_clusters * d, 0.0f);
    std::vector<size_t> cs_under_fp(n_clusters, 0);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t k = run_fp.q_assignments[i];
        if (k >= n_clusters) continue;
        ++cs_under_fp[k];
        float* row = c_true_fp.data() + static_cast<size_t>(k) * d;
        for (size_t j = 0; j < d; ++j) row[j] += data[i * d + j];
    }
    for (size_t k = 0; k < n_clusters; ++k) {
        if (cs_under_fp[k] == 0) continue;
        const float inv = 1.0f / static_cast<float>(cs_under_fp[k]);
        float* row = c_true_fp.data() + k * d;
        for (size_t j = 0; j < d; ++j) row[j] *= inv;
    }
    auto norm_true_fp = NormFromMean(c_true_fp.data(), global_mean, n_clusters, d);
    std::vector<double> true_obs_fp_err(n_clusters);
    for (size_t k = 0; k < n_clusters; ++k) {
        true_obs_fp_err[k] = std::abs(norm_true_fp[k] - norm_fp[k]);
    }
    PrintDistribution("sanity: |‖true_fp-m‖-‖obs_fp-m‖|", true_obs_fp_err);

    // Per-cluster predicted inflation: ‖c_pred − m‖ / ‖c_true_local − m‖
    std::vector<double> predicted_cluster_inflation(n_clusters, 1.0);
    for (size_t k = 0; k < n_clusters; ++k) {
        if (norm_true_local[k] > 0.0)
            predicted_cluster_inflation[k] = norm_pred[k] / norm_true_local[k];
    }
    PrintDistribution("predicted cluster inflation    ", predicted_cluster_inflation);

    // Observed cluster inflation: ‖c_q − m‖ / ‖c_fp − m‖ (already computed as `ratio`)
    std::vector<double> observed_cluster_inflation = ratio;
    PrintDistribution("observed cluster inflation     ", observed_cluster_inflation);

    std::cout << "  pearson (pred_inflation, obs_inflation) = " << std::setprecision(4)
              << Pearson(predicted_cluster_inflation, observed_cluster_inflation) << "\n"
              << "  spearman(pred_inflation, obs_inflation) = "
              << Spearman(predicted_cluster_inflation, observed_cluster_inflation) << "\n";

    std::cout << "\n  Top 10 largest clusters under FP=false (the 'magnets'):\n"
              << "    rank | centroid | size  | ||c-m||_q  | ||c-m||_fp | shrink\n";
    std::vector<size_t> by_size(n_clusters);
    std::iota(by_size.begin(), by_size.end(), 0u);
    std::sort(by_size.begin(), by_size.end(),
              [&](size_t a, size_t b) { return sizes_q[a] > sizes_q[b]; });
    for (size_t r = 0; r < std::min<size_t>(10, n_clusters); ++r) {
        const size_t c = by_size[r];
        std::cout << "    " << std::setw(4) << (r + 1)
                  << " | " << std::setw(8) << c
                  << " | " << std::setw(5) << sizes_q[c]
                  << " | " << std::setw(10) << std::fixed << std::setprecision(3) << norm_q[c]
                  << " | " << std::setw(10) << norm_fp[c]
                  << " | " << std::setw(6) << std::setprecision(3) << ratio[c] << "\n";
    }

    return 0;
}
