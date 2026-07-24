#!/bin/bash

# End-to-end benchmark runner for all algorithms
# Usage: ./end_to_end.sh [-b build_dir] [-p python_cmd] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   -p python_cmd: Python command to use (default: python3)
#   -t seconds:    Per-run timeout in seconds (default: 3600 = 1 hour; 0 disables)
#   datasets: Dataset names (default: mxbai openai)
#
# Examples:
#   ./end_to_end.sh                              # Run all datasets with default build dir and python3
#   ./end_to_end.sh mxbai openai                 # Run only mxbai and openai
#   ./end_to_end.sh -b ../build mxbai            # Run mxbai with custom build dir
#   ./end_to_end.sh -p /path/to/python mxbai     # Run mxbai with custom Python

set -e

BUILD_DIR="../cmake-build-release"
PYTHON_CMD="python3"
RUN_TIMEOUT=3600   # per-run timeout in seconds (1 hour); 0 disables

while getopts "b:p:t:" opt; do
    case $opt in
        b)
            BUILD_DIR="$OPTARG"
            ;;
        p)
            PYTHON_CMD="$OPTARG"
            ;;
        t)
            RUN_TIMEOUT="$OPTARG"
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
echo "End-to-End Benchmark Suite"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Project root: $PROJECT_ROOT"
echo "Python command: $PYTHON_CMD"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""
echo "Building C++ benchmarks..."
cd "$BUILD_DIR_ABS"
cmake --build . --target end_to_end_superkmeans.out end_to_end_faiss.out end_to_end_hierarchical.out end_to_end_fast_kmeans.out end_to_end_marigold.out end_to_end_daskmeans.out -j
echo "Build complete!"
echo ""

cd "$SCRIPT_DIR"

# Pick a timeout command: GNU `timeout` (Linux) or `gtimeout` (macOS + coreutils).
TIMEOUT_CMD=""
if [ "$RUN_TIMEOUT" -gt 0 ]; then
    if command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD="timeout"
    elif command -v gtimeout >/dev/null 2>&1; then
        TIMEOUT_CMD="gtimeout"
    else
        echo "WARNING: no 'timeout'/'gtimeout' found — per-run timeout disabled" \
             "(on macOS: brew install coreutils)"
    fi
fi

# Run one benchmark under the per-run timeout. Never aborts the whole suite:
# a timeout (exit 124/137) or failure is reported and we move on. -k gives a
# stuck process 60s after SIGTERM before SIGKILL.
run_bench() {
    local rc=0
    if [ -n "$TIMEOUT_CMD" ]; then
        "$TIMEOUT_CMD" -k 60 "$RUN_TIMEOUT" "$@" || rc=$?
    else
        "$@" || rc=$?
    fi
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        echo ">>> TIMEOUT: killed after ${RUN_TIMEOUT}s: $*"
    elif [ "$rc" -ne 0 ]; then
        echo ">>> FAILED (exit $rc): $*"
    fi
    return 0
}

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "########################################## "
    echo "# DATASET: $DATASET"
    echo "########################################## "
    echo ""
    echo "=========================================="
    echo "Running benchmarks for $DATASET..."
    echo "=========================================="
    # echo ""
    # echo "----------------------------------------"
    # echo "1/4: SuperKMeans"
    # echo "----------------------------------------"
    # run_bench "$BUILD_DIR_ABS/benchmarks/end_to_end_superkmeans.out" "$DATASET"
    # echo ""
    # echo "----------------------------------------"
    # echo "2/4: Hierarchical SuperKMeans"
    # echo "----------------------------------------"
    # run_bench "$BUILD_DIR_ABS/benchmarks/end_to_end_hierarchical.out" "$DATASET"
    # echo ""
    # echo "----------------------------------------"
    # echo "3/4: FAISS Clustering"
    # echo "----------------------------------------"
    # run_bench "$BUILD_DIR_ABS/benchmarks/end_to_end_faiss.out" "$DATASET"
    # echo ""
    # echo "----------------------------------------"
    # echo "4/4: scikit-learn KMeans"
    # echo "----------------------------------------"
    # run_bench "$PYTHON_CMD" end_to_end/end_to_end_scikit.py "$DATASET"
    # echo ""
    # --- Competitors (run selectively; can be slow — Marigold and especially
    # --- Dask-means degrade to near-serial on high-dimensional embeddings) ---
    run_bench "$BUILD_DIR_ABS/benchmarks/end_to_end_fast_kmeans.out" "$DATASET" beta_hamerly
    # run_bench "$BUILD_DIR_ABS/benchmarks/end_to_end_fast_kmeans.out" "$DATASET" beta_kmeans
    run_bench "$BUILD_DIR_ABS/benchmarks/end_to_end_marigold.out" "$DATASET"
    run_bench "$BUILD_DIR_ABS/benchmarks/end_to_end_daskmeans.out" "$DATASET"
done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "Results written to: $SCRIPT_DIR/results/\$SKM_ARCH/end_to_end.csv"
echo "  (where \$SKM_ARCH=${SKM_ARCH:-default})"
