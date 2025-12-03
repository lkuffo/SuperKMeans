#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "superkmeans/pdx/adsampling.h"

using namespace skmeans;

int main(int argc, char** argv) {
    // Test multiple dimensionalities
    std::vector<size_t> dims_to_test = {768, 1024, 1536};  // Non-power-of-2

    if (argc > 1) {
        dims_to_test = {static_cast<size_t>(std::atoi(argv[1]))};
    }

    for (size_t num_dims : dims_to_test) {
        std::cout << "\n=== Testing dimensionality: " << num_dims << " ===" << std::endl;

        constexpr size_t n_vectors = 100;
        std::vector<float> original(n_vectors * num_dims);
        std::vector<float> rotated(n_vectors * num_dims);
        std::vector<float> unrotated(n_vectors * num_dims);

        // Generate random data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (size_t i = 0; i < original.size(); ++i) {
            original[i] = dist(gen);
        }

        // Create pruner (which initializes the rotation)
        ADSamplingPruner<Quantization::f32> pruner(num_dims, 2.1f, 42);

        // Rotate
        std::cout << "Rotating..." << std::endl;
        pruner.Rotate(original.data(), rotated.data(), n_vectors);

        // Check that rotation changed the data
        bool data_changed = false;
        for (size_t i = 0; i < 10; ++i) {
            if (std::abs(rotated[i] - original[i]) > 1e-5f) {
                data_changed = true;
                break;
            }
        }
        if (!data_changed) {
            std::cout << "⚠ WARNING: Rotation didn't change the data!" << std::endl;
        }

        // Print first few values after rotation
        std::cout << "First 5 rotated values: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << rotated[i] << " ";
        }
        std::cout << std::endl;

        // Check for NaN or inf in rotated data
        bool has_invalid = false;
        for (size_t i = 0; i < rotated.size(); ++i) {
            if (std::isnan(rotated[i]) || std::isinf(rotated[i])) {
                has_invalid = true;
                std::cout << "✗ INVALID VALUE at index " << i << ": " << rotated[i] << std::endl;
                break;
            }
        }
        if (has_invalid) {
            std::cout << "✗ Rotation produced NaN/Inf - FAILED" << std::endl;
            continue;
        }

        // Unrotate
        std::cout << "Unrotating..." << std::endl;
        pruner.Unrotate(rotated.data(), unrotated.data(), n_vectors);

        // Print first few values after unrotation
        std::cout << "First 5 unrotated values: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << unrotated[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "First 5 original values: ";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << original[i] << " ";
        }
        std::cout << std::endl;

        // Verify correctness
        float max_error = 0.0f;
        float avg_error = 0.0f;
        size_t n_errors = 0;

        for (size_t i = 0; i < original.size(); ++i) {
            float error = std::abs(unrotated[i] - original[i]);
            avg_error += error;
            if (error > max_error) {
                max_error = error;
            }
            if (error > 1e-3f) {
                n_errors++;
                if (n_errors <= 10) {
                    std::cout << "Large error at index " << i
                              << " (vec " << (i / num_dims) << ", dim " << (i % num_dims) << "): "
                              << "original=" << original[i] << ", "
                              << "unrotated=" << unrotated[i] << ", "
                              << "error=" << error << std::endl;
                }
            }
        }

        avg_error /= original.size();

        std::cout << "Max error: " << max_error << std::endl;
        std::cout << "Avg error: " << avg_error << std::endl;
        std::cout << "Errors > 1e-3: " << n_errors << " / " << original.size() << std::endl;

        if (max_error < 1e-3f) {
            std::cout << "✓ DCT rotation is CORRECT for dim=" << num_dims << std::endl;
        } else {
            std::cout << "✗ DCT rotation is INCORRECT for dim=" << num_dims << std::endl;
            return 1;
        }
    }

    return 0;
}
