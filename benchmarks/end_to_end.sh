#!/bin/bash

# End-to-end benchmark runner for all algorithms
# Usage: ./end_to_end.sh [-b build_dir] [-p python_cmd] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   -p python_cmd: Python command to use (default: python3)
#   datasets: Dataset names (default: mxbai openai wiki arxiv sift fmnist glove200 glove100 glove50 gist contriever)
#
# Examples:
#   ./end_to_end.sh                              # Run all datasets with default build dir and python3
#   ./end_to_end.sh mxbai openai                 # Run only mxbai and openai
#   ./end_to_end.sh -b ../build mxbai            # Run mxbai with custom build dir
#   ./end_to_end.sh -p /path/to/python mxbai     # Run mxbai with custom Python

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
    DATASETS=(mxbai openai wiki arxiv sift fmnist glove200 gist contriever)
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
echo "End-to-End Benchmark Suite"
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
cmake --build . --target end_to_end_superkmeans.out end_to_end_faiss.out -j
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
    "$BUILD_DIR_ABS/benchmarks/end_to_end_superkmeans.out" "$DATASET"
    echo ""

    # 2. FAISS (C++)
    echo "----------------------------------------"
    echo "2/3: FAISS Clustering"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/end_to_end_faiss.out" "$DATASET"
    echo ""

    # 3. scikit-learn KMeans (Python)
    echo "----------------------------------------"
    echo "3/3: scikit-learn KMeans"
    echo "----------------------------------------"
    "$PYTHON_CMD" end_to_end/end_to_end_scikit.py "$DATASET"
    echo ""

done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "Results written to: $SCRIPT_DIR/results/\$SKM_ARCH/end_to_end.csv"
echo "  (where \$SKM_ARCH=${SKM_ARCH:-default})"
