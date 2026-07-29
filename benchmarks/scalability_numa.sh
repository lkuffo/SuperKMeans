#!/bin/bash

# NUMA-aware thread-scaling sweep for the revision (Section 4.6 / Figure 11).
#
# Replaces the manual "deactivate cores in AWS" workflow: the thread count is set in
# software via OMP_NUM_THREADS (the benchmark reads omp_get_max_threads()), and NUMA
# memory/CPU placement is set via numactl. The script runs the full
# (placement x thread-count) grid, so no AWS knobs need touching.
#
# Placement configs (each writes its own CSV: results/$SKM_ARCH/scalability_numa_<cfg>.csv):
#   node0      : CPUs + memory pinned to NUMA node 0        (single-node baseline, no NUMA)
#   interleave : memory interleaved across all NUMA nodes   (NUMA-aware; both nodes)
#   naive      : default first-touch (local) allocation     (both nodes; shows the penalty)
#
# The n_threads column in each CSV distinguishes the points within a curve.
#
# Usage: ./scalability_numa.sh [-b build_dir] [-a "algos"] [-c "configs"] [-t "threads"] [datasets...]
#   -b  build dir                              (default ../build)
#   -a  algorithms: superkmeans faiss scikit   (default: superkmeans)
#   -c  configs:    node0 interleave naive      (default: node0 interleave naive)
#   -t  thread counts                          (default: 1 2 4 8 16 24 32 48 64 96)
#   datasets                                   (default: mxbai openai)
#
# Requires: numactl, lscpu (standard on AWS Linux). PYTHON env overrides the scikit interpreter.

set -e

BUILD_DIR="../build"
ALGOS="superkmeans"
CONFIGS="node0 interleave naive"
THREADS_LIST="1 2 4 8 16 24 32 48 64 96"
PYTHON="${PYTHON:-python3}"

while getopts "b:a:c:t:" opt; do
    case $opt in
        b) BUILD_DIR="$OPTARG" ;;
        a) ALGOS="$OPTARG" ;;
        c) CONFIGS="$OPTARG" ;;
        t) THREADS_LIST="$OPTARG" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND-1))

DATASETS=( "$@" )
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(mxbai openai)

command -v numactl >/dev/null 2>&1 || { echo "ERROR: numactl not found (e.g. sudo apt-get install -y numactl)"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$BUILD_DIR" = /* ]]; then
    BUILD_DIR_ABS="$BUILD_DIR"
else
    BUILD_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$BUILD_DIR" && pwd)"
fi

# ---- topology ----
NUMA_NODES=$(numactl -H | awk '/^available:/{print $2}')
PHYS_CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{gsub(/ /,"",c);gsub(/ /,"",s);print c*s}')
VCPUS=$(nproc)
[ -z "$NUMA_NODES" ] && NUMA_NODES=1
PER_NODE_PHYS=$(( PHYS_CORES / NUMA_NODES ))

echo "=========================================="
echo "NUMA scalability sweep"
echo "=========================================="
echo "NUMA nodes=$NUMA_NODES  physical cores=$PHYS_CORES  vCPUs=$VCPUS  cores/node=$PER_NODE_PHYS"
echo "------------------------------------------"
numactl -H | sed 's/^/  /'
echo "------------------------------------------"
echo "algos=[$ALGOS]  configs=[$CONFIGS]  threads=[$THREADS_LIST]  datasets=[${DATASETS[*]}]"
echo "results -> results/${SKM_ARCH:-default}/scalability_numa_<config>.csv"
echo "=========================================="

# ---- build ----
TARGETS=""
[[ "$ALGOS" == *superkmeans* ]] && TARGETS="$TARGETS end_to_end_superkmeans.out"
[[ "$ALGOS" == *faiss* ]] && TARGETS="$TARGETS end_to_end_faiss.out"
if [ -n "$TARGETS" ]; then
    echo "Building:$TARGETS"
    ( cd "$BUILD_DIR_ABS" && cmake --build . --target $TARGETS -j )
fi
cd "$SCRIPT_DIR"

cap_for()     { case "$1" in node0) echo "$PER_NODE_PHYS" ;; *) echo "$VCPUS" ;; esac; }
numactl_for() { case "$1" in
                    node0)      echo "--cpunodebind=0 --membind=0" ;;
                    interleave) echo "--interleave=all" ;;
                    naive)      echo "--localalloc" ;;
                    *) echo "" ;;
                esac; }

run_one() {  # algo dataset config n_threads numactl_args...
    local algo="$1" ds="$2" cfg="$3" N="$4"; shift 4
    local exp="scalability_numa_${cfg}"
    local bin
    case "$algo" in
        superkmeans) bin=("$BUILD_DIR_ABS/benchmarks/end_to_end_superkmeans.out" "$ds" "$exp") ;;
        faiss)       bin=("$BUILD_DIR_ABS/benchmarks/end_to_end_faiss.out" "$ds" "$exp") ;;
        scikit)      bin=("$PYTHON" end_to_end/end_to_end_scikit.py "$ds" "$exp") ;;
        *) echo "  unknown algo: $algo"; return 1 ;;
    esac
    OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N \
    OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
        numactl "$@" "${bin[@]}"
}

set +e  # keep going if a single run fails
for ds in "${DATASETS[@]}"; do
    for cfg in $CONFIGS; do
        CAP=$(cap_for "$cfg")
        read -r -a NC <<< "$(numactl_for "$cfg")"
        echo ""; echo "### dataset=$ds  config=$cfg  (numactl ${NC[*]}; cap=$CAP threads)"
        for N in $THREADS_LIST; do
            [ "$N" -gt "$CAP" ] && continue
            [ "$N" -gt "$PHYS_CORES" ] && smt="  [SMT: >physical cores]" || smt=""
            for algo in $ALGOS; do
                echo ">>> $ds | $cfg | threads=$N | $algo$smt"
                run_one "$algo" "$ds" "$cfg" "$N" "${NC[@]}"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Done. CSVs in results/${SKM_ARCH:-default}/ :"
for cfg in $CONFIGS; do echo "  scalability_numa_${cfg}.csv"; done
echo "=========================================="
