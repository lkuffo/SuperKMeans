#!/bin/bash

# Early termination benchmark runner for algorithms
# Usage: ./early_termination.sh [-b build_dir] [-p python_cmd] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   -p python_cmd: Python command to use (default: python3)
#   datasets: Dataset names (default: mxbai wiki openai arxiv sift fmnist glove100 glove50 gist contriever)
#
# Examples:
#   ./early_termination.sh                              # Run all datasets with default build dir and python3
#   ./early_termination.sh mxbai openai                 # Run only mxbai and openai
#   ./early_termination.sh -b ../build mxbai            # Run mxbai with custom build dir
#   ./early_termination.sh -p /path/to/python mxbai     # Run mxbai with custom Python

set -e  # Exit on error

# Default build directory and Python command
BUILD_DIR="../cmake-build-release"
PYTHON_CMD="python3"

# Parse flags
while getopts "b:p:" opt; do
    case $opt in
        b)
            BUILD_DIR="$OPTARG"
            ;;
        p)
            PYTHON_CMD="$OPTARG"
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
echo "Early Termination Benchmark Suite"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Project root: $PROJECT_ROOT"
echo "Python command: $PYTHON_CMD"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""

# Build C++ benchmarks
echo "Building C++ benchmarks..."
cd "$BUILD_DIR_ABS"
cmake --build . --target early_termination_superkmeans.out early_termination_faiss.out -j
echo "Build complete!"
echo ""

# Change to benchmarks directory so Python scripts can find data files
cd "$SCRIPT_DIR"

# Loop over datasets
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "########################################## "
    echo "# DATASET: $DATASET"
    echo "########################################## "
    echo ""

    # Run benchmarks
    echo "=========================================="
    echo "Running benchmarks for $DATASET..."
    echo "=========================================="
    echo ""

    # 1. SuperKMeans (C++)
    echo "----------------------------------------"
    echo "1/3: SuperKMeans"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/early_termination_superkmeans.out" "$DATASET"
    echo ""

    # 2. FAISS (C++) - runs twice internally with different n_iters
    echo "----------------------------------------"
    echo "2/3: FAISS Clustering"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/early_termination_faiss.out" "$DATASET"
    echo ""

    # 3. scikit-learn KMeans (Python)
    echo "----------------------------------------"
    echo "3/3: scikit-learn KMeans"
    echo "----------------------------------------"
    "$PYTHON_CMD" early_termination/early_termination_scikit.py "$DATASET"
    echo ""
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "Results written to: $SCRIPT_DIR/results/\$SKM_ARCH/early_termination.csv"
echo "  (where \$SKM_ARCH=${SKM_ARCH:-default})"
