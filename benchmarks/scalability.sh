#!/bin/bash

# Scalability benchmark runner for all algorithms
# Usage: ./scalability.sh [-b build_dir] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   datasets: Dataset names (default: mxbai openai)
#
# Examples:
#   ./scalability.sh                      # Run all datasets with default build dir
#   ./scalability.sh mxbai openai         # Run only mxbai and openai
#   ./scalability.sh -b ../build mxbai    # Run mxbai with custom build dir

set -e

BUILD_DIR="../cmake-build-release"

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
echo "Scalability Benchmark Suite"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Project root: $PROJECT_ROOT"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""
echo "Building C++ benchmarks..."
cd "$BUILD_DIR_ABS"
cmake --build . --target end_to_end_superkmeans.out end_to_end_faiss.out -j
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
    "$BUILD_DIR_ABS/benchmarks/end_to_end_superkmeans.out" "$DATASET" "scalability"
    echo ""
    echo "----------------------------------------"
    echo "2/3: FAISS Clustering"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/end_to_end_faiss.out" "$DATASET" "scalability"
    echo ""
    echo "----------------------------------------"
    echo "3/3: scikit-learn KMeans"
    echo "----------------------------------------"
    python3 end_to_end/end_to_end_scikit.py "$DATASET" "scalability"
    echo ""
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "Results written to: $SCRIPT_DIR/results/\$SKM_ARCH/scalability.csv"
echo "  (where \$SKM_ARCH=${SKM_ARCH:-default})"
