#!/bin/bash

# Pareto benchmark runner for SuperKMeans (grid search over n_iters and sampling_fraction)
# Usage: ./pareto.sh [-b build_dir] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   datasets: Dataset names (default: mxbai wiki openai arxiv sift fmnist glove100 glove50 gist contriever)
#
# Examples:
#   ./pareto.sh                      # Run all datasets with default build dir
#   ./pareto.sh mxbai openai         # Run only mxbai and openai
#   ./pareto.sh -b ../build mxbai    # Run mxbai with custom build dir

set -e  # Exit on error

# Default build directory
BUILD_DIR="../cmake-build-release"

# Parse flags
while getopts "b:" opt; do
    case $opt in
        b)
            BUILD_DIR="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Shift past the flags
shift $((OPTIND-1))

# Define datasets array
if [ $# -gt 0 ]; then
    # Use datasets from command line arguments
    DATASETS=("$@")
else
    # Default datasets
    DATASETS=(mxbai openai wiki arxiv sift fmnist glove200 glove100 glove50 gist contriever)
fi

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve BUILD_DIR to absolute path
if [[ "$BUILD_DIR" = /* ]]; then
    # Already absolute
    BUILD_DIR_ABS="$BUILD_DIR"
else
    # Relative to SCRIPT_DIR (where this script is located)
    BUILD_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$BUILD_DIR" && pwd)"
fi

echo "=========================================="
echo "Pareto Benchmark Suite (Grid Search)"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Project root: $PROJECT_ROOT"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""

# Build C++ benchmark
echo "Building C++ benchmark..."
cd "$BUILD_DIR_ABS"
cmake --build . --target pareto_superkmeans.out -j
echo "Build complete!"
echo ""

# Change to benchmarks directory so scripts can find data files
cd "$SCRIPT_DIR"

# Loop over datasets
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "########################################## "
    echo "# DATASET: $DATASET"
    echo "########################################## "
    echo ""

    # Run benchmark
    echo "=========================================="
    echo "Running pareto benchmark for $DATASET..."
    echo "=========================================="
    echo ""

    # SuperKMeans (C++) - Grid search over n_iters and sampling_fraction
    echo "----------------------------------------"
    echo "SuperKMeans Grid Search"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/pareto_superkmeans.out" "$DATASET"
    echo ""
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "Results written to: $SCRIPT_DIR/results/\$SKM_ARCH/pareto.csv"
echo "  (where \$SKM_ARCH=${SKM_ARCH:-default})"
