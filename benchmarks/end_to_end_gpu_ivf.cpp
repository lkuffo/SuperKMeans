#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <faiss/utils/utils.h>

#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/IndexFlat.h>

#include "bench_utils.h"
#include "superkmeans/nanobench.h"

int main(int argc, char* argv[]) {
    // Experiment configuration
    const std::string algorithm = "faiss_gpu_ivf_flat_cuvs";

    std::string dataset = (argc > 1) ? std::string(argv[1]) : std::string("openai");

    // Experiment name can be passed as second argument (default: "end_to_end_gpu_ivf")
    std::string experiment_name = (argc > 2) ? std::string(argv[2]) : std::string("end_to_end_gpu_ivf");

    auto it = bench_utils::DATASET_PARAMS.find(dataset);
    if (it == bench_utils::DATASET_PARAMS.end()) {
        std::cerr << "Unknown dataset '" << dataset << "'\n";
        std::cerr << "Known datasets: mxbai, openai, wiki, arxiv, sift, fmnist\n";
        return 1;
    }

    const size_t n = it->second.first;
    const size_t d = it->second.second;
    const size_t n_clusters =
        std::max<int>(1u, static_cast<int>(std::sqrt(static_cast<double>(n)) * 4.0));
    int n_iters = bench_utils::MAX_ITERS;
    const size_t THREADS = omp_get_max_threads();
    omp_set_num_threads(THREADS);
    std::string filename = bench_utils::get_data_path(dataset);

    std::cout << "=== Running algorithm: " << algorithm << " ===" << std::endl;
    std::cout << "Dataset: " << dataset << " (n=" << n << ", d=" << d << ")\n";
    std::cout << "n_clusters: " << n_clusters << std::endl;
    std::cout << "Compile options: " << faiss::get_compile_options() << std::endl;

    std::vector<float> data;
    try {
        data.resize(n * d);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate data vector for n*d = " << (n * d) << ": " << e.what()
                  << "\n";
        return 1;
    }
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    std::cout << "Data loaded successfully" << std::endl;

    // Initialize GPU resources
    faiss::gpu::StandardGpuResources res;

    // Create a CPU quantizer index for training
    faiss::IndexFlatL2 cpu_quantizer(d);

    // Set up IVF-Flat index configuration
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = 0;
    config.use_cuvs = true;  // Enable cuvs backend

    // Create the GPU IVF-Flat index
    std::cout << "\n--- Building IVF-Flat Index ---" << std::endl;
    std::cout << "n_lists (clusters): " << n_clusters << std::endl;
    std::cout << "n_iters: " << n_iters << std::endl;
    std::cout << "use_cuvs: true" << std::endl;

    faiss::gpu::GpuIndexIVFFlat index(
        &res,
        d,
        n_clusters,
        faiss::METRIC_L2,
        config
    );

    // Set clustering parameters
    index.cp.niter = n_iters;
    index.cp.max_points_per_centroid = 999999; // We don't want to take samples
    index.cp.verbose = false;

    // Check if this dataset should use angular/spherical k-means
    auto is_angular = std::find(
        bench_utils::ANGULAR_DATASETS.begin(),
        bench_utils::ANGULAR_DATASETS.end(),
        dataset
    );
    if (is_angular != bench_utils::ANGULAR_DATASETS.end()) {
        std::cout << "Using spherical k-means for dataset: " << dataset << std::endl;
        index.cp.spherical = true;
    }

    // Time the index training (which includes clustering)
    bench_utils::TicToc timer;
    timer.Tic();
    index.train(n, data.data());
    timer.Toc();
    double construction_time_ms = timer.GetMilliseconds();

    // Get actual iterations - note: detailed stats may not be available with cuvs backend
    int actual_iterations = n_iters;  // Assumed to complete all iterations
    double final_objective = -1.0;  // Not directly available from GPU index

    std::cout << "\nTraining completed in " << construction_time_ms << " ms" << std::endl;
    std::cout << "Actual iterations: " << actual_iterations << " (requested: " << n_iters << ")"
              << std::endl;
    std::cout << "Final objective: " << final_objective << " (not available)" << std::endl;

    // Create config dictionary with FAISS GPU IVF-Flat parameters
    std::unordered_map<std::string, std::string> config_map;
    config_map["n_lists"] = std::to_string(n_clusters);
    config_map["niter"] = std::to_string(index.cp.niter);
    config_map["max_points_per_centroid"] = std::to_string(index.cp.max_points_per_centroid);
    config_map["min_points_per_centroid"] = std::to_string(index.cp.min_points_per_centroid);
    config_map["seed"] = std::to_string(index.cp.seed);
    config_map["spherical"] = index.cp.spherical ? "true" : "false";
    config_map["verbose"] = index.cp.verbose ? "true" : "false";
    config_map["use_cuvs"] = config.use_cuvs ? "true" : "false";
    config_map["device"] = std::to_string(config.device);

    // Empty recall results (recall computation skipped)
    std::vector<std::tuple<int, float, float, float, float>> results_knn_10;
    std::vector<std::tuple<int, float, float, float, float>> results_knn_100;

    std::cout << "\nNote: Recall computation skipped for GPU IVF-Flat benchmark" << std::endl;
    std::cout << "Writing results to CSV..." << std::endl;

    // Write results to CSV
    bench_utils::write_results_to_csv(
        experiment_name,
        algorithm,
        dataset,
        n_iters,
        actual_iterations,
        d,
        n,
        n_clusters,
        construction_time_ms,
        static_cast<int>(THREADS),
        final_objective,
        config_map,
        results_knn_10,
        results_knn_100
    );

    std::cout << "Benchmark complete!" << std::endl;
    return 0;
}