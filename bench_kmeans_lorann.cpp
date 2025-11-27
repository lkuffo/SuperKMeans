#define ANKERL_NANOBENCH_IMPLEMENT
#define EIGEN_USE_THREADS

#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

#include "superkmeans/nanobench.h"
#include "superkmeans/pdx/layout.h"
#include "superkmeans/pdx/adsampling.h"
#include "superkmeans/superkmeans.h"
#include "superkmeans/pdx/utils.h"

#include "lorann/lorann/lorann.h"


int main(int argc, char* argv[]) {
    std::cout << "Compiles!" << std::endl;

    // SKMeans
    const int n = 262144;
    const int d = 1024;
    int n_clusters = 1024;
    int n_iters = 1;
    float sampling_fraction = 1.0;

    std::vector<skmeans::skmeans_value_t<skmeans::f32>> data(n * d);
    std::ifstream file(std::string{CMAKE_SOURCE_DIR} + "/data_random.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << std::endl;
        return 1;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    omp_set_num_threads(10);
    auto km = Lorann::KMeans(n_clusters, n_iters);
    ankerl::nanobench::Bench().epochs(1).epochIterations(1).run("Lorann KMeans", [&]() {
        km.train(data.data(), n, d, true, 10);
    });

}
