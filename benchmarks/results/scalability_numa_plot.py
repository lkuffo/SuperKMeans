#!/usr/bin/env python3
"""Plot NUMA thread-scaling from scalability_numa_<config>.csv.

Speedup vs thread count, one line per placement config (node0 / interleave / naive),
one panel per dataset. Overlays the ideal-linear line and, if a clock ratio is given,
a clock-adjusted ceiling  speedup(n) = n * (all_core_clock / single_core_clock)  that
explains sub-linearity as frequency throttling rather than NUMA. Vertical guides mark
the NUMA-node boundary (where the 2nd node engages) and where SMT begins.

Usage:
  python3 benchmarks/results/scalability_numa_plot.py [arch] [--algo superkmeans]
      [--clock-ratio 0.57] [--node-boundary 24] [--smt-boundary 48] [--datasets mxbai openai]
"""
import argparse, csv, os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

CONFIGS = [  # (csv suffix, legend label, colour, marker)
    ("node0",      "single NUMA node",         "#1f77b4", "o"),
    ("interleave", "interleaved (both nodes)", "#2ca02c", "s"),
    ("naive",      "first-touch (both nodes)", "#d62728", "^"),
]


def load(arch, algo):
    base = f"benchmarks/results/{arch}"
    data = defaultdict(lambda: defaultdict(dict))   # data[cfg][dataset][threads] = min time_ms
    for cfg, *_ in CONFIGS:
        path = f"{base}/scalability_numa_{cfg}.csv"
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if r.get("algorithm") != algo:
                continue
            ds = r["dataset"]; N = int(r["threads"]); t = float(r["construction_time_ms"])
            d = data[cfg][ds]
            d[N] = min(t, d.get(N, float("inf")))    # dedup re-runs -> best time
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", nargs="?", default="r8i")
    ap.add_argument("--algo", default="superkmeans")
    ap.add_argument("--clock-ratio", type=float, default=None,
                    help="all-core / single-core clock ratio (e.g. 0.57) for the ceiling line")
    ap.add_argument("--node-boundary", type=int, default=None,
                    help="threads at the NUMA-node boundary (default: max of the node0 sweep)")
    ap.add_argument("--smt-boundary", type=int, default=None,
                    help="threads where SMT begins = physical core count (default: 2x node boundary)")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    data = load(args.arch, args.algo)
    if not data:
        print(f"no scalability_numa_*.csv (algo={args.algo}) under benchmarks/results/{args.arch}")
        return

    dss = args.datasets or sorted({ds for cfg in data for ds in data[cfg]})
    nb = args.node_boundary
    if nb is None and data.get("node0"):
        nb = max((max(v) for v in data["node0"].values()), default=None)
    smt = args.smt_boundary or (2 * nb if nb else None)

    fig, axes = plt.subplots(1, len(dss), figsize=(7 * len(dss), 5.8), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        t1s = [data[cfg][ds][1] for cfg, *_ in CONFIGS
               if cfg in data and ds in data[cfg] and 1 in data[cfg][ds]]
        if not t1s:
            ax.set_title(f"{ds}: no 1-thread baseline"); continue
        T1 = sorted(t1s)[len(t1s) // 2]              # shared baseline = median 1-thread time
        maxN = 1
        for cfg, label, color, mk in CONFIGS:
            if cfg not in data or ds not in data[cfg]:
                continue
            pts = sorted(data[cfg][ds].items())
            xs = [n for n, _ in pts]; ys = [T1 / t for _, t in pts]
            maxN = max(maxN, max(xs))
            ax.plot(xs, ys, marker=mk, color=color, lw=1.4, ms=5.5, label=label)

        ax.plot([1, maxN], [1, maxN], ls=":", color="0.55", lw=1.0, label="linear (ideal)")
        if args.clock_ratio:
            ax.plot([1, maxN], [args.clock_ratio, args.clock_ratio * maxN], ls="--", color="0.35",
                    lw=1.0, label=f"clock-adjusted (×{args.clock_ratio:g})")

        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xticks([t for t in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96) if t <= maxN])
        ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlim(0.9, maxN * 1.15)

        y0 = ax.get_ylim()[0]
        for x, c, txt in ((nb, "purple", "NUMA node 2"), (smt, "0.4", "SMT")):
            if x and 1 < x <= maxN:
                ax.axvline(x, color=c, ls="-.", lw=0.9, alpha=0.6)
                ax.text(x, y0 * 1.15, f" {txt}", rotation=90, fontsize=7.5, color=c, va="bottom")

        ax.set_xlabel("threads (cores)")
        ax.set_ylabel(rf"speedup  $T(1)/T(n)$   ($T(1)$={T1/1000:.1f}s)")
        ax.set_title(ds)
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{args.algo} thread scaling on {args.arch} (2-node NUMA) — speedup vs threads",
                 fontsize=12)
    out = f"benchmarks/results/{args.arch}/scalability_numa"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=140, bbox_inches="tight")
    print(f"wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
