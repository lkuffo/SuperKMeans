#!/bin/bash

# NUMA-aware thread-scaling sweep for the revision (Section 4.6 / Figure 11).
#
# Replaces the manual "deactivate cores in AWS" workflow. Instead of physically
# offlining cores, each run is pinned to exactly N logical CPUs with
# `numactl --physcpubind`, which is the software equivalent: both OpenMP and the
# threaded BLAS see N cores and use N threads, with no oversubscription.
#
# IMPORTANT: we do NOT set OMP_PROC_BIND / OMP_PLACES. On x86 the f32 path calls a
# top-level threaded BLAS (sgemm) alongside separate OpenMP regions; setting
# OMP_PROC_BIND makes the BLAS worker threads inherit the master's pinned single-core
# affinity mask, so all N BLAS threads pile onto one core and the run gets ~N x SLOWER
# as N grows. The cpuset is controlled by numactl only.
#
# Placement configs (each writes results/$SKM_ARCH/scalability_numa_<cfg>.csv):
#   node0      : N cores on NUMA node 0, memory on node 0             (single-node baseline)
#   interleave : N cores filling node0 then node1, memory interleaved (NUMA-aware headline)
#   naive      : same N cores, first-touch (local) memory             (remote-access penalty)
# Physical cores are filled first (node0 then node1); SMT siblings only for N > phys cores.
# The n_threads column in each CSV distinguishes the points within a curve.
#
# Usage: ./scalability_numa.sh [-b build_dir] [-a "algos"] [-c "configs"] [-t "threads"] [datasets...]
#   -b build dir (default ../build)   -a algos (default superkmeans)   -c configs (default interleave; node0/naive selectable)
#   -t thread counts (default 96..1 descending, so the biggest run goes first)   datasets (default mxbai openai)
# Requires numactl + lscpu. PYTHON env overrides the scikit interpreter.

set -e

BUILD_DIR="../build"
ALGOS="superkmeans"
CONFIGS="interleave"
THREADS_LIST="96 64 48 32 24 16 8 4 2 1"
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

command -v numactl >/dev/null 2>&1 || { echo "ERROR: numactl not found"; exit 1; }
command -v lscpu  >/dev/null 2>&1 || { echo "ERROR: lscpu not found"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$BUILD_DIR" = /* ]]; then BUILD_DIR_ABS="$BUILD_DIR"
else BUILD_DIR_ABS="$(cd "$SCRIPT_DIR" && cd "$BUILD_DIR" && pwd)"; fi

# ---- topology ----
NUMA_NODES=$(numactl -H | awk '/^available:/{print $2}'); [ -z "$NUMA_NODES" ] && NUMA_NODES=1
PHYS_CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{gsub(/ /,"",c);gsub(/ /,"",s);print c*s}')
VCPUS=$(nproc)
PER_NODE_PHYS=$(( PHYS_CORES / NUMA_NODES ))

# ordered logical CPUs: physical cores first (node-ordered), then SMT siblings
CORE_ORDER=$(lscpu -p=CPU,CORE,NODE | grep -v '^#' | sort -t, -k3,3n -k2,2n -k1,1n | awk -F, '
  { key=$3","$2; if(!(key in seen)){seen[key]=1; prim[++np]=$1} else {sec[++ns]=$1} }
  END { for(i=1;i<=np;i++) print prim[i]; for(i=1;i<=ns;i++) print sec[i] }')
CORE_ARR=($CORE_ORDER)

echo "=========================================="
echo "NUMA scalability sweep (cpuset-pinned; no OMP_PROC_BIND)"
echo "NUMA nodes=$NUMA_NODES  physical cores=$PHYS_CORES  vCPUs=$VCPUS  cores/node=$PER_NODE_PHYS"
echo "core fill order (first 12): ${CORE_ARR[*]:0:12} ..."
numactl -H | sed 's/^/  /'
echo "algos=[$ALGOS]  configs=[$CONFIGS]  threads=[$THREADS_LIST]  datasets=[${DATASETS[*]}]"
echo "results -> results/${SKM_ARCH:-default}/scalability_numa_<config>.csv"
echo "=========================================="

TARGETS=""
[[ "$ALGOS" == *superkmeans* ]] && TARGETS="$TARGETS end_to_end_superkmeans.out"
[[ "$ALGOS" == *faiss* ]] && TARGETS="$TARGETS end_to_end_faiss.out"
if [ -n "$TARGETS" ]; then
    echo "Building:$TARGETS"; ( cd "$BUILD_DIR_ABS" && cmake --build . --target $TARGETS -j )
fi
cd "$SCRIPT_DIR"

cap_for() { case "$1" in node0) echo "$PER_NODE_PHYS" ;; *) echo "$VCPUS" ;; esac; }
mem_for() { case "$1" in node0) echo "--membind=0" ;; interleave) echo "--interleave=all" ;; naive) echo "--localalloc" ;; *) echo "--localalloc" ;; esac; }

run_one() {  # algo dataset config n_threads cpuset mempolicy
    local algo="$1" ds="$2" cfg="$3" N="$4" cpus="$5" mem="$6"
    local exp="scalability_numa_${cfg}" bin
    case "$algo" in
        superkmeans) bin=("$BUILD_DIR_ABS/benchmarks/end_to_end_superkmeans.out" "$ds" "$exp") ;;
        faiss)       bin=("$BUILD_DIR_ABS/benchmarks/end_to_end_faiss.out" "$ds" "$exp") ;;
        scikit)      bin=("$PYTHON" end_to_end/end_to_end_scikit.py "$ds" "$exp") ;;
        *) echo "  unknown algo: $algo"; return 1 ;;
    esac
    # thread COUNT only (no OMP_PROC_BIND/OMP_PLACES); placement via the numactl cpuset.
    OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N \
        numactl --physcpubind="$cpus" $mem "${bin[@]}"
}

set +e  # keep going if a single run fails
for ds in "${DATASETS[@]}"; do
    for cfg in $CONFIGS; do
        CAP=$(cap_for "$cfg"); MEM=$(mem_for "$cfg")
        echo ""; echo "### dataset=$ds  config=$cfg  ($MEM; cap=$CAP threads)"
        for N in $THREADS_LIST; do
            [ "$N" -gt "$CAP" ] && continue
            CPUS=$(IFS=,; echo "${CORE_ARR[*]:0:$N}")
            [ "$N" -gt "$PHYS_CORES" ] && smt="  [SMT]" || smt=""
            for algo in $ALGOS; do
                echo ">>> $ds | $cfg | threads=$N | $algo$smt  (cpus=$CPUS)"
                run_one "$algo" "$ds" "$cfg" "$N" "$CPUS" "$MEM"
            done
        done
    done
done

echo ""; echo "Done. CSVs in results/${SKM_ARCH:-default}/ :"
for cfg in $CONFIGS; do echo "  scalability_numa_${cfg}.csv"; done
