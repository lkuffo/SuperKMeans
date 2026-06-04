#!/bin/bash

# Accelerators benchmark runner — exhaustive parameter sweep
# Usage: ./accelerators.sh [-b build_dir] [dataset1] [dataset2] ...
#   -b build_dir: Build directory (default: ../cmake-build-release)
#   datasets: Dataset names (default: mxbai openai)
#
# Examples:
#   ./accelerators.sh                          # Run all datasets with default build dir
#   ./accelerators.sh mxbai openai             # Run only mxbai and openai
#   ./accelerators.sh -b ../build mxbai        # Run mxbai with custom build dir

set -e

BUILD_DIR="../cmake-build-release"

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

BIN="$BUILD_DIR_ABS/benchmarks/accelerators.out"

echo "=========================================="
echo "Accelerators Benchmark Suite"
echo "=========================================="
echo "Build directory: $BUILD_DIR_ABS"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="
echo ""

echo "Building accelerators.out..."
cd "$BUILD_DIR_ABS"
cmake --build . --target accelerators.out -j
echo "Build complete!"
echo ""

cd "$SCRIPT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Naming convention for the arguments:
#   $BIN <dataset> <dim_reduction> <quantizer> <quantized_centroid_update>
#        <full_precision_final_centroids> <use_blas_only>
#
# Sensible combinations:
#   f32:    quantized_centroid_update and full_precision_final_centroids are N/A
#           → always false/false, only vary use_blas_only
#   sq8/sq4/rabitq:
#           full_precision_final_centroids only matters when quantized_centroid_update=true
#           → skip (quantized_centroid_update=false, full_precision_final_centroids=true)
# ─────────────────────────────────────────────────────────────────────────────

STEP=0

run() {
    STEP=$((STEP+1))
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "── [$STEP] $1 ──"
    shift
    "$BIN" "$@"
}

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# DATASET: $DATASET"
    echo "######################################################################"

    # ==================================================================
    #  RAW (no dimensionality reduction)
    # ==================================================================

    # ── raw + hnsw (no pruning support — blas-only path) ──
    run "raw / f32 / hnsw"                                          "$DATASET" raw hnsw false false true
    run "raw / sq8 / hnsw_usearch"                                  "$DATASET" raw hnsw_sq8 false false true
    run "raw / sq8 / hnsw_faiss"                                    "$DATASET" raw hnsw_faiss_sq8 false false true

    # ── hierarchical superkmeans (raw only; f32 and sq8) ──
    run "raw / f32 / hsk / blas-only"                               "$DATASET" hsk f32 false false true
    run "raw / sq8 / hsk / blas-only"                               "$DATASET" hsk sq8 true false true
    run "raw / f32 / hsk / pruning"                               "$DATASET" hsk f32 false false true
    run "raw / sq8 / hsk / pruning"                               "$DATASET" hsk sq8 true false true

    continue

    # ── raw + f32 ──
    run "raw / f32 / pruning"                                       "$DATASET" raw f32 false false false
    run "raw / f32 / blas-only"                                     "$DATASET" raw f32 false false true

    # ── raw + sq8 ──
    # run "raw / sq8 / no-quant-update / pruning"                     "$DATASET" raw sq8 false false false
    run "raw / sq8 / quant-update / pruning"                        "$DATASET" raw sq8 true false false
    # run "raw / sq8 / quant-update + full-prec-final / pruning"      "$DATASET" raw sq8 true true false
    # run "raw / sq8 / no-quant-update / blas-only"                   "$DATASET" raw sq8 false false true
    run "raw / sq8 / quant-update / blas-only"                      "$DATASET" raw sq8 true false true
    # run "raw / sq8 / quant-update + full-prec-final / blas-only"    "$DATASET" raw sq8 true true true

    # ── raw + sq4 ──
    # run "raw / sq4 / no-quant-update / pruning"                     "$DATASET" raw sq4 false false false
    run "raw / sq4 / quant-update / pruning"                        "$DATASET" raw sq4 true false false
    run "raw / sq4 / quant-update + full-prec-final / pruning"      "$DATASET" raw sq4 true true false
    # run "raw / sq4 / no-quant-update / blas-only"                   "$DATASET" raw sq4 false false true
    run "raw / sq4 / quant-update / blas-only"                      "$DATASET" raw sq4 true false true
    run "raw / sq4 / quant-update + full-prec-final / blas-only"    "$DATASET" raw sq4 true true true

    # ── raw + lvq4 ──
    # run "raw / lvq4 / no-quant-update / pruning"                     "$DATASET" raw lvq4 false false false
    run "raw / lvq4 / quant-update / pruning"                        "$DATASET" raw lvq4 true false false
    run "raw / lvq4 / quant-update + full-prec-final / pruning"      "$DATASET" raw lvq4 true true false
    # run "raw / lvq4 / no-quant-update / blas-only"                   "$DATASET" raw lvq4 false false true
    run "raw / lvq4 / quant-update / blas-only"                      "$DATASET" raw lvq4 true false true
    run "raw / lvq4 / quant-update + full-prec-final / blas-only"    "$DATASET" raw lvq4 true true true

    # # ── raw + rabitq ──
    run "raw / rabitq / quant-update / pruning"                   "$DATASET" raw rabitq true false false
    run "raw / rabitq / quant-update + full-prec-final / pruning" "$DATASET" raw rabitq true true false
    # run "raw / rabitq / no-quant-update / blas-only"                "$DATASET" raw rabitq false false true
    run "raw / rabitq / quant-update / blas-only"                   "$DATASET" raw rabitq true false true
    run "raw / rabitq / quant-update + full-prec-final / blas-only" "$DATASET" raw rabitq true true true

    # ==================================================================
    #  PCA (dimensionality reduction, iterates over TARGET_D internally)
    # ==================================================================

    # ── pca + f32 ──
    run "pca / f32 / pruning"                                       "$DATASET" pca f32 false false false
    run "pca / f32 / blas-only"                                     "$DATASET" pca f32 false false true

    # ── pca + sq8 ──
    # run "pca / sq8 / no-quant-update / pruning"                     "$DATASET" pca sq8 false false false
    run "pca / sq8 / quant-update / pruning"                        "$DATASET" pca sq8 true false false
    # run "pca / sq8 / quant-update + full-prec-final / pruning"      "$DATASET" pca sq8 true true false
    # run "pca / sq8 / no-quant-update / blas-only"                   "$DATASET" pca sq8 false false true
    run "pca / sq8 / quant-update / blas-only"                      "$DATASET" pca sq8 true false true
    # run "pca / sq8 / quant-update + full-prec-final / blas-only"    "$DATASET" pca sq8 true true true

    # ── pca + sq4 ──
    # run "pca / sq4 / no-quant-update / pruning"                     "$DATASET" pca sq4 false false false
    run "pca / sq4 / quant-update / pruning"                        "$DATASET" pca sq4 true false false
    # run "pca / sq4 / quant-update + full-prec-final / pruning"      "$DATASET" pca sq4 true true false
    # run "pca / sq4 / no-quant-update / blas-only"                   "$DATASET" pca sq4 false false true
    run "pca / sq4 / quant-update / blas-only"                      "$DATASET" pca sq4 true false true
    # run "pca / sq4 / quant-update + full-prec-final / blas-only"    "$DATASET" pca sq4 true true true

    # ── pca + lvq4 ──
    # run "pca / lvq4 / no-quant-update / pruning"                     "$DATASET" pca lvq4 false false false
    run "pca / lvq4 / quant-update / pruning"                        "$DATASET" pca lvq4 true false false
    # run "pca / lvq4 / quant-update + full-prec-final / pruning"      "$DATASET" pca lvq4 true true false
    # run "pca / lvq4 / no-quant-update / blas-only"                   "$DATASET" pca lvq4 false false true
    run "pca / lvq4 / quant-update / blas-only"                      "$DATASET" pca lvq4 true false true
    # run "pca / lvq4 / quant-update + full-prec-final / blas-only"    "$DATASET" pca lvq4 true true true

    # ── pca + rabitq ──
    run "pca / rabitq / quant-update / pruning"                   "$DATASET" pca rabitq true false false
    run "pca / rabitq / quant-update + full-prec-final / pruning" "$DATASET" pca rabitq true true false
    # run "pca / rabitq / no-quant-update / blas-only"                "$DATASET" pca rabitq false false true
    run "pca / rabitq / quant-update / blas-only"                   "$DATASET" pca rabitq true false true
    run "pca / rabitq / quant-update + full-prec-final / blas-only" "$DATASET" pca rabitq true true true

    # ==================================================================
    #  JLT (dimensionality reduction, iterates over TARGET_D internally)
    #  Deactivated with Quantization as PCA always perform better
    # ==================================================================

    # ── jlt + f32 ──
    run "jlt / f32 / pruning"                                       "$DATASET" jlt f32 false false false
    run "jlt / f32 / blas-only"                                     "$DATASET" jlt f32 false false true

    # # ── jlt + sq8 ──
    # run "jlt / sq8 / no-quant-update / pruning"                     "$DATASET" jlt sq8 false false false
    # run "jlt / sq8 / quant-update / pruning"                        "$DATASET" jlt sq8 true false false
    # run "jlt / sq8 / quant-update + full-prec-final / pruning"      "$DATASET" jlt sq8 true true false
    # run "jlt / sq8 / no-quant-update / blas-only"                   "$DATASET" jlt sq8 false false true
    # run "jlt / sq8 / quant-update / blas-only"                      "$DATASET" jlt sq8 true false true
    # run "jlt / sq8 / quant-update + full-prec-final / blas-only"    "$DATASET" jlt sq8 true true true

    # # ── jlt + sq4 ──
    # run "jlt / sq4 / no-quant-update / pruning"                     "$DATASET" jlt sq4 false false false
    # run "jlt / sq4 / quant-update / pruning"                        "$DATASET" jlt sq4 true false false
    # run "jlt / sq4 / quant-update + full-prec-final / pruning"      "$DATASET" jlt sq4 true true false
    # run "jlt / sq4 / no-quant-update / blas-only"                   "$DATASET" jlt sq4 false false true
    # run "jlt / sq4 / quant-update / blas-only"                      "$DATASET" jlt sq4 true false true
    # run "jlt / sq4 / quant-update + full-prec-final / blas-only"    "$DATASET" jlt sq4 true true true

    # # ── jlt + lvq4 ──
    # run "jlt / lvq4 / no-quant-update / pruning"                     "$DATASET" jlt lvq4 false false false
    # run "jlt / lvq4 / quant-update / pruning"                        "$DATASET" jlt lvq4 true false false
    # run "jlt / lvq4 / quant-update + full-prec-final / pruning"      "$DATASET" jlt lvq4 true true false
    # run "jlt / lvq4 / no-quant-update / blas-only"                   "$DATASET" jlt lvq4 false false true
    # run "jlt / lvq4 / quant-update / blas-only"                      "$DATASET" jlt lvq4 true false true
    # run "jlt / lvq4 / quant-update + full-prec-final / blas-only"    "$DATASET" jlt lvq4 true true true

    # # ── jlt + rabitq (no pruning support — blas-only) ──
    # run "jlt / rabitq / no-quant-update / blas-only"                "$DATASET" jlt rabitq false false true
    # run "jlt / rabitq / quant-update / blas-only"                   "$DATASET" jlt rabitq true false true
    # run "jlt / rabitq / quant-update + full-prec-final / blas-only" "$DATASET" jlt rabitq true true true

    # ==================================================================
    #  Matryoshka (truncation, no quantizer combinations)
    #  Deactivated with Quantization as PCA always perform better
    # ==================================================================

    # ── mat + f32 ──
    run "mat / f32 / pruning"                                       "$DATASET" mat f32 false false false
    run "mat / f32 / blas-only"                                     "$DATASET" mat f32 false false true

    # ==================================================================
    #  PQ (Product Quantization — raw only, sparse voting, blas-only)
    #  M values iterated internally: {8, 16, 32, 64, 96, 128}
    # ==================================================================

    # ── raw + pq8 (8-bit PQ, Ks=256, scalar SDC) ──
    run "raw / pq8 / sparse-voting / blas-only / M sweep"                        "$DATASET" raw pq8 true false true
    # run "raw / pq8 / sparse-voting + full-prec-final / blas-only / M sweep"      "$DATASET" raw pq8 true true true

    # ── raw + pq4 (4-bit PQ, Ks=16, FastScan SIMD) ──
    run "raw / pq4 / sparse-voting / blas-only / M sweep"                        "$DATASET" raw pq4 true false true
    run "raw / pq4 / sparse-voting + full-prec-final / blas-only / M sweep"      "$DATASET" raw pq4 true true true

done

echo ""
echo "=========================================="
echo "All accelerators benchmarks complete! ($STEP runs)"
echo "=========================================="
echo ""
echo "CSV files written to: $SCRIPT_DIR/results/${SKM_ARCH:-default}/accelerators_*.csv"
