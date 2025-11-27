/**
 * Simple test to verify Rotate followed by Unrotate gives back the original vectors.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>

#include "superkmeans/common.h"
#include "superkmeans/pdx/adsampling.h"

int main() {
    // Test both low-dim (orthonormal matrix) and high-dim (DCT) cases
    std::vector<size_t> dimensions = {50, 128, 256, 512, 768, 1024};
    
    for (size_t d : dimensions) {
        std::cout << "\n=== Testing d=" << d << " ===" << std::endl;
        std::cout << "Using " << (d >= skmeans::D_THRESHOLD_FOR_DCT_ROTATION ? "DCT" : "orthonormal matrix") << " rotation" << std::endl;
        
        const size_t n = 100;  // Number of test vectors
        
        // Create random test vectors
        std::vector<float> original(n * d);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < n * d; ++i) {
            original[i] = dist(rng);
        }
        
        // Create pruner (which has Rotate/Unrotate)
        skmeans::ADSamplingPruner<skmeans::f32> pruner(d, 2.1f);
        
        // Rotate
        std::vector<float> rotated(n * d);
        pruner.Rotate(original.data(), rotated.data(), n);
        
        // Unrotate
        std::vector<float> recovered(n * d);
        pruner.Unrotate(rotated.data(), recovered.data(), n);
        
        // Compare original and recovered
        double max_error = 0.0;
        double sum_error = 0.0;
        for (size_t i = 0; i < n * d; ++i) {
            double error = std::abs(original[i] - recovered[i]);
            max_error = std::max(max_error, error);
            sum_error += error;
        }
        double avg_error = sum_error / (n * d);
        
        std::cout << "Max error: " << max_error << std::endl;
        std::cout << "Avg error: " << avg_error << std::endl;
        
        // Check if recovered values approximately match original
        bool passed = max_error < 1e-4;
        if (passed) {
            std::cout << "✓ PASSED" << std::endl;
        } else {
            std::cout << "✗ FAILED - error too large!" << std::endl;
            
            // Print first few mismatches
            std::cout << "First 5 mismatches:" << std::endl;
            int count = 0;
            for (size_t i = 0; i < n * d && count < 5; ++i) {
                double error = std::abs(original[i] - recovered[i]);
                if (error > 1e-4) {
                    std::cout << "  [" << i << "] original=" << original[i] 
                              << ", recovered=" << recovered[i] 
                              << ", error=" << error << std::endl;
                    count++;
                }
            }
        }
    }
    
    return 0;
}

