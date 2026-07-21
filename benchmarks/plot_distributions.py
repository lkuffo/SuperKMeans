# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "h5py",
#     "pandas",
#     "seaborn",
#     "matplotlib",
# ]
# ///
"""Per-dimension KDE cloud for a fixed sample of a dataset, each dimension standardized.

Every dimension is z-scored (mean 0, std 1) then drawn as one KDE curve, all overlaid.

Usage: uv run benchmarks/plot_distributions.py <dataset> [n_sample]
Saves: benchmarks/results/distributions/<dataset>.png
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils

OUT_DIR = bench_utils.BENCHMARKS_ROOT / "results" / "distributions"
N_SAMPLE = 10000


def plot_kde(data, **kwargs) -> "matplotlib.axes.Axes":
    g = sns.kdeplot(
        data,
        fill=True,
        palette="mako",  # crest
        linewidth=0.04,
        alpha=0.005,
        log_scale=False,
        **kwargs
    )
    g.set_title("")
    g.set_xlabel("")
    g.set_ylabel("")
    g.set_xticklabels([])
    g.set_yticklabels([])
    g.axis('off')
    g.legend().set_visible(False)
    return g


def standardize(sample):
    sd = sample.std(axis=0)
    keep = sd > 0
    return (sample[:, keep] - sample[:, keep].mean(axis=0)) / sd[keep]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <dataset> [n_sample]")
    dataset = sys.argv[1]
    n, d = bench_utils.DATASET_PARAMS[dataset]
    n_sample = int(sys.argv[2]) if len(sys.argv) >= 3 else N_SAMPLE
    n_sample = min(n_sample, n)

    X = np.fromfile(bench_utils.get_data_path(dataset), dtype=np.float32, count=n * d).reshape(n, d)
    rng = np.random.default_rng(42)
    sample = X[rng.choice(n, n_sample, replace=False)]

    plt.figure(figsize=(6, 4))
    plot_kde(pd.DataFrame(standardize(sample)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{dataset}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {out}  ({n_sample} vectors x {d} dims)")


if __name__ == "__main__":
    main()