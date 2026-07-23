#!/bin/bash

# Rotation microbenchmark runner: current extra-buffer rotation vs in-place rotation
# Usage: ./rotation.sh [-b build_dir] [-r reps] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   -r reps: Timed repetitions per approach (default: benchmark default)
#   datasets: Dataset names (default: mxbai)
#
# Examples:
#   ./rotation.sh                        # Run mxbai with default build dir
#   ./rotation.sh mxbai openai           # Run mxbai and openai
#   ./rotation.sh -b ../build -r 5 mxbai # Custom build dir and 5 reps

set -e

BUILD_DIR="../cmake-build-release"
REPS=""

while getopts "b:r:" opt; do
    case $opt in
        b)
            BUILD_DIR="$OPTARG"
            ;;
        r)
            REPS="$OPTARG"
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
    DATASETS=(mxbai)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$BUILD_DIR" = /* ]]; then
    BUILD_DIR_ABS="$BUILD_DIR"
else
    BUILD_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$BUILD_DIR" && pwd)"
fi

echo "=========================================="
echo "Rotation Microbenchmark"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""
echo "Building rotation microbenchmark..."
cd "$BUILD_DIR_ABS"
cmake --build . --target microbenchmark_rotation.out -j
echo "Build complete!"
echo ""

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "########################################## "
    echo "# DATASET: $DATASET"
    echo "########################################## "
    echo ""
    "$BUILD_DIR_ABS/benchmarks/microbenchmark_rotation.out" "$DATASET" $REPS
    echo ""
done
