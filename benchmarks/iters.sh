#!/bin/bash

# Iterations benchmark runner for SuperKMeans + FAISS.
# Runs both, iterations 1-10 on the full dataset, recording the fine recall /
# vectors_explored grid (0.10..20.00). Outputs iters_superkmeans.csv and iters_faiss.csv.
#
# Usage: ./iters.sh [-b build_dir] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../build)
#   datasets: Dataset names (default: mxbai openai)

set -e

BUILD_DIR="../build"

while getopts "b:" opt; do
    case $opt in
        b) BUILD_DIR="$OPTARG" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(mxbai openai)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$BUILD_DIR" = /* ]]; then
    BUILD_DIR_ABS="$BUILD_DIR"
else
    BUILD_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$BUILD_DIR" && pwd)"
fi

echo "=========================================="
echo "Iterations Benchmark Suite (SuperKMeans + FAISS)"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""
echo "Building C++ benchmarks..."
cd "$BUILD_DIR_ABS"
cmake --build . --target iters_superkmeans.out iters_faiss.out -j
echo "Build complete!"
echo ""

cd "$SCRIPT_DIR"

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "########################################## "
    echo "# DATASET: $DATASET"
    echo "########################################## "
    echo ""
    echo "----------------------------------------"
    echo "1/2: SuperKMeans (iterations 1-10)"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/iters_superkmeans.out" "$DATASET"
    echo ""
    echo "----------------------------------------"
    echo "2/2: FAISS (iterations 1-10)"
    echo "----------------------------------------"
    "$BUILD_DIR_ABS/benchmarks/iters_faiss.out" "$DATASET"
    echo ""
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo "Results written to: $SCRIPT_DIR/results/\$SKM_ARCH/iters_{superkmeans,faiss}.csv"
echo "  (where \$SKM_ARCH=${SKM_ARCH:-default})"
