#!/bin/bash

# Varying_k benchmark runner for all algorithms
# Usage: ./varying_k.sh [-b build_dir] [-p python_cmd] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   -p python_cmd: Python command to use (default: python3)
#   datasets: Dataset names (default: mxbai openai)
#
# Examples:
#   ./varying_k.sh                              # Run all datasets with default build dir and python3
#   ./varying_k.sh mxbai openai                 # Run only mxbai and openai
#   ./varying_k.sh -b ../build mxbai            # Run mxbai with custom build dir
#   ./varying_k.sh -p /path/to/python mxbai     # Run mxbai with custom Python

set -e

BUILD_DIR="../cmake-build-release"
PYTHON_CMD="python3"

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

shift $((OPTIND-1))

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(mxbai openai)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "$BUILD_DIR" = /* ]]; then
    BUILD_DIR_ABS="$BUILD_DIR"
else
    BUILD_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$BUILD_DIR" && pwd)"
fi

echo "=========================================="
echo "Varying K Benchmark Suite"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Project root: $PROJECT_ROOT"
echo "Python command: $PYTHON_CMD"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""
echo "Building C++ benchmarks..."
cd "$BUILD_DIR_ABS"
cmake --build . --target varying_k_superkmeans.out varying_k_faiss.out -j
echo "Build complete!"
echo ""

cd "$SCRIPT_DIR"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "########################################## "
    echo "# DATASET: $DATASET"
    echo "########################################## "
    echo ""
    echo "=========================================="
    echo "Running benchmarks for $DATASET..."
    echo "=========================================="
    echo ""
    echo "----------------------------------------"
    echo "1/3: SuperKMeans"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/varying_k_superkmeans.out" "$DATASET"
    echo ""
    echo "----------------------------------------"
    echo "2/3: FAISS Clustering"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/varying_k_faiss.out" "$DATASET"
    echo ""
    echo "----------------------------------------"
    echo "3/3: scikit-learn KMeans"
    echo "----------------------------------------"
    "$PYTHON_CMD" varying_k/varying_k_scikit.py "$DATASET"
    echo ""
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "Results written to: $SCRIPT_DIR/results/\$SKM_ARCH/varying_k.csv"
echo "  (where \$SKM_ARCH=${SKM_ARCH:-default})"
