#!/bin/bash

# Iterations benchmark runner for scikit-learn
# Usage: ./iters_scikit.sh [-p python_cmd] [dataset1] [dataset2] ...
#   -p python_cmd: Python command to use (default: python3)
#   datasets: Dataset names (default: mxbai openai wiki arxiv sift fmnist glove200 glove100 glove50 gist contriever)
#
# Examples:
#   ./iters_scikit.sh                              # Run all datasets with python3
#   ./iters_scikit.sh mxbai openai                 # Run only mxbai and openai
#   ./iters_scikit.sh -p /path/to/python mxbai     # Run mxbai with custom Python

set -e  # Exit on error

# Default Python command
PYTHON_CMD="python3"

# Parse flags
while getopts "p:" opt; do
    case $opt in
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

echo "=========================================="
echo "Iterations Benchmark Suite (scikit-learn)"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo "Python command: $PYTHON_CMD"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
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

    # Run benchmark
    echo "=========================================="
    echo "Running iterations benchmark for $DATASET..."
    echo "=========================================="
    echo ""

    # scikit-learn KMeans (Python) with varying iterations
    echo "----------------------------------------"
    echo "scikit-learn KMeans (iterations: 1-10, init: random and k-means++)"
    echo "----------------------------------------"
    "$PYTHON_CMD" iters/iters_scikit.py "$DATASET"
    echo ""

done

echo ""
echo "=========================================="
echo "All benchmarks complete!"
echo "=========================================="
echo ""
echo "Results written to: $SCRIPT_DIR/results/\$SKM_ARCH/iters_scikit.csv"
echo "  (where \$SKM_ARCH=${SKM_ARCH:-default})"
