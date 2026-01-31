#!/bin/bash

# Pareto benchmark runner for SuperKMeans (grid search over n_iters and sampling_fraction)
# Usage: ./pareto.sh [-b build_dir] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   datasets: Dataset names (default: mxbai openai)
#
# Examples:
#   ./pareto.sh                      # Run all datasets with default build dir
#   ./pareto.sh mxbai openai         # Run only mxbai and openai
#   ./pareto.sh -b ../build mxbai    # Run mxbai with custom build dir

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
echo "Pareto Benchmark Suite (Grid Search)"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Project root: $PROJECT_ROOT"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""
echo "Building C++ benchmark..."
cd "$BUILD_DIR_ABS"
cmake --build . --target pareto_superkmeans.out -j
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
    echo "Running pareto benchmark for $DATASET..."
    echo "=========================================="
    echo ""
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
