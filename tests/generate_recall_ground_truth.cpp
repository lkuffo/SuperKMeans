#undef HAS_FFTW

#include <cstdio>
#include <string>

#include "recall_utils.h"

int main() {
    using namespace skm_test;
    const std::string path = CMAKE_SOURCE_DIR "/tests/test_data.bin";

    const float f32 =
        ClusteringRecall<skmeans::Quantization::f32>(skmeans::QuantizerType::none, path);
    const float sq8 =
        ClusteringRecall<skmeans::Quantization::u8>(skmeans::QuantizerType::sq8, path);
    const float lvq4 =
        ClusteringRecall<skmeans::Quantization::u8>(skmeans::QuantizerType::lvq4, path);
    const float rabitq =
        ClusteringRecall<skmeans::Quantization::u8>(skmeans::QuantizerType::rabitq, path);
    const float hier_f32 = HierarchicalClusteringRecall<skmeans::Quantization::f32>(
        skmeans::QuantizerType::none, path
    );
    const float hier_sq8 =
        HierarchicalClusteringRecall<skmeans::Quantization::u8>(skmeans::QuantizerType::sq8, path);
    const float hier_lvq4 =
        HierarchicalClusteringRecall<skmeans::Quantization::u8>(skmeans::QuantizerType::lvq4, path);
    const float hier_rabitq = HierarchicalClusteringRecall<skmeans::Quantization::u8>(
        skmeans::QuantizerType::rabitq, path
    );

    std::fprintf(
        stderr,
        "\nrecall@%d @ %.0f%% clusters (n_clusters=%zu, d=%zu, %zu queries)\n",
        RECALL_KNN,
        RECALL_FRAC * 100.0f,
        RECALL_N_CLUSTERS,
        RECALL_D,
        RECALL_N_QUERIES
    );
    std::fprintf(stderr, "f32=%.3f  sq8=%.3f  lvq4=%.3f  rabitq=%.3f\n", f32, sq8, lvq4, rabitq);
    std::fprintf(
        stderr,
        "hierarchical: f32=%.3f  sq8=%.3f  lvq4=%.3f  rabitq=%.3f\n",
        hier_f32,
        hier_sq8,
        hier_lvq4,
        hier_rabitq
    );
    std::fprintf(
        stderr,
        "degradation vs f32:  sq8=%.3f  lvq4=%.3f  rabitq=%.3f\n",
        f32 - sq8,
        f32 - lvq4,
        f32 - rabitq
    );

    std::printf(
        "// recall@%d @ %.0f%% clusters. Regenerate with generate_recall_ground_truth.out\n",
        RECALL_KNN,
        RECALL_FRAC * 100.0f
    );
    std::printf("f32:                 %.3ff\n", f32);
    std::printf("sq8:                 %.3ff\n", sq8);
    std::printf("lvq4:                %.3ff\n", lvq4);
    std::printf("rabitq:              %.3ff\n", rabitq);
    std::printf("hierarchical_f32:    %.3ff\n", hier_f32);
    std::printf("hierarchical_sq8:    %.3ff\n", hier_sq8);
    std::printf("hierarchical_lvq4:   %.3ff\n", hier_lvq4);
    std::printf("hierarchical_rabitq: %.3ff\n", hier_rabitq);
    return 0;
}
