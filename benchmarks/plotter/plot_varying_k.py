#!/usr/bin/env python3
"""
Plot varying_k benchmark results for a specific dataset.

Creates publication-quality plots showing SuperKMeans speedup over baselines.
Uses Droid Serif font and vibrant colors matching scientific paper style.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


# Vibrant color palette matching the reference image
COLORS = {
    'superkmeans': '#27ae60',  # Vibrant green
    'faiss': '#e74c3c',        # Vibrant red
    'scikit': '#9b59b6',       # Vibrant purple
}


def setup_paper_style():
    """Configure matplotlib for publication-quality plots with Droid Serif font."""
    # Try to use Droid Serif, fallback to serif if not available
    try:
        # Check if Droid Serif is available
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'Droid Serif' in available_fonts:
            plt.rcParams['font.family'] = 'Droid Serif'
        else:
            # Fallback to other serif fonts
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman']
            print("Warning: Droid Serif not found, using fallback serif font")
    except Exception as e:
        print(f"Warning: Could not set font: {e}")
        plt.rcParams['font.family'] = 'serif'

    # Publication quality settings
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 15


def parse_config(config_str):
    """Parse the config JSON string."""
    try:
        config_str = config_str.replace('""', '"')
        return json.loads(config_str)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing config: {e}")
        return None


def plot_speedup_bars(df_dataset, dataset_name, output_dir):
    """
    Create a grouped bar chart showing construction time and speedup.

    Shows SuperKMeans, FAISS, and scikit-learn side by side for each k value,
    with speedup annotations above each group.
    """
    setup_paper_style()

    # Group by algorithm and n_clusters
    algo_data = {}
    for algo in ['superkmeans', 'faiss', 'scikit']:
        df_algo = df_dataset[df_dataset['algorithm'] == algo]
        if len(df_algo) > 0:
            algo_data[algo] = df_algo.sort_values('n_clusters')

    if 'superkmeans' not in algo_data:
        print(f"No SuperKMeans data found for {dataset_name}")
        return

    # Get k values
    k_values = sorted(df_dataset['n_clusters'].unique())

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                     gridspec_kw={'height_ratios': [3, 2]})

    # === Subplot 1: Construction time bars ===
    x = np.arange(len(k_values))
    width = 0.25

    # Get SuperKMeans times for speedup calculation
    skm_times = algo_data['superkmeans'].set_index('n_clusters')['construction_time_ms']

    for i, (algo, label) in enumerate([('superkmeans', 'SuperKMeans'),
                                        ('faiss', 'FAISS'),
                                        ('scikit', 'scikit-learn')]):
        if algo in algo_data:
            times = []
            for k in k_values:
                df_k = algo_data[algo][algo_data[algo]['n_clusters'] == k]
                if len(df_k) > 0:
                    times.append(df_k['construction_time_ms'].values[0])
                else:
                    times.append(0)

            offset = (i - 1) * width
            bars = ax1.bar(x + offset, times, width, label=label,
                          color=COLORS.get(algo, '#95a5a6'),
                          edgecolor='black', linewidth=1.2, alpha=0.85)

            # Add value labels on bars
            for bar, time in zip(bars, times):
                if time > 0:
                    height = bar.get_height()
                    # Format time nicely
                    if time < 1000:
                        label = f'{time:.0f}'
                    elif time < 10000:
                        label = f'{time/1000:.1f}k'
                    else:
                        label = f'{time/1000:.0f}k'

                    ax1.text(bar.get_x() + bar.get_width() / 2., height,
                            label, ha='center', va='bottom',
                            fontsize=8, fontweight='normal')

    # Add speedup annotations above each group
    for i, k in enumerate(k_values):
        skm_time = skm_times.get(k, None)
        if skm_time is None or skm_time == 0:
            continue

        # Calculate max height for this group
        max_height = 0
        speedups = []

        for algo in ['faiss', 'scikit']:
            if algo in algo_data:
                df_k = algo_data[algo][algo_data[algo]['n_clusters'] == k]
                if len(df_k) > 0:
                    time = df_k['construction_time_ms'].values[0]
                    max_height = max(max_height, time)
                    speedup = time / skm_time
                    speedups.append(f'{speedup:.1f}x')

        if speedups:
            speedup_text = ' / '.join(speedups)
            ax1.text(i, max_height * 1.08, speedup_text,
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    color='#2c3e50')

    ax1.set_ylabel('Construction Time (ms)', fontweight='bold')
    ax1.set_title(f'K-Means Construction Time - {dataset_name}\n' +
                  f'n={df_dataset.iloc[0]["data_size"]:,}, d={df_dataset.iloc[0]["dimensionality"]}',
                  fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{k:,}' for k in k_values])
    ax1.legend(loc='upper left', framealpha=0.95, edgecolor='black', fancybox=False)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    # === Subplot 2: Speedup relative to SuperKMeans ===
    for algo, label in [('faiss', 'FAISS vs SuperKMeans'),
                        ('scikit', 'scikit-learn vs SuperKMeans')]:
        if algo in algo_data:
            speedups = []
            for k in k_values:
                skm_time = skm_times.get(k, None)
                df_k = algo_data[algo][algo_data[algo]['n_clusters'] == k]

                if skm_time and len(df_k) > 0:
                    time = df_k['construction_time_ms'].values[0]
                    speedup = time / skm_time
                    speedups.append(speedup)
                else:
                    speedups.append(None)

            # Plot line with markers
            valid_indices = [i for i, s in enumerate(speedups) if s is not None]
            valid_k = [k_values[i] for i in valid_indices]
            valid_speedups = [speedups[i] for i in valid_indices]

            ax2.plot(valid_k, valid_speedups, 'o-',
                    label=label, color=COLORS.get(algo, '#95a5a6'),
                    linewidth=2.5, markersize=8, alpha=0.85)

            # Add speedup value labels
            for k, speedup in zip(valid_k, valid_speedups):
                ax2.text(k, speedup, f'{speedup:.1f}x',
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold')

    # Add baseline line at 1.0x
    ax2.axhline(y=1.0, color='#27ae60', linestyle='--',
                linewidth=2, alpha=0.7, label='SuperKMeans (baseline)')

    ax2.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax2.set_ylabel('Speedup Factor\n(Baseline Time / SuperKMeans Time)', fontweight='bold')
    ax2.set_title('SuperKMeans Speedup Analysis', fontweight='bold', pad=10)
    ax2.set_xscale('log')
    ax2.set_xticks(k_values)
    ax2.set_xticklabels([f'{k:,}' for k in k_values])
    ax2.legend(loc='best', framealpha=0.95, edgecolor='black', fancybox=False)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_axisbelow(True)
    ax2.set_ylim(bottom=0.5)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'{dataset_name}_varying_k_speedup.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_time_scaling(df_dataset, dataset_name, output_dir):
    """
    Create a log-log plot showing how construction time scales with k.

    Similar style to the compression ratio plot in the reference image.
    """
    setup_paper_style()

    # Group by algorithm
    algo_data = {}
    for algo in ['superkmeans', 'faiss', 'scikit']:
        df_algo = df_dataset[df_dataset['algorithm'] == algo]
        if len(df_algo) > 0:
            algo_data[algo] = df_algo.sort_values('n_clusters')

    if not algo_data:
        print(f"No data found for {dataset_name}")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each algorithm
    markers = {'superkmeans': 'o', 'faiss': 's', 'scikit': '^'}
    labels = {'superkmeans': 'SuperKMeans', 'faiss': 'FAISS', 'scikit': 'scikit-learn'}

    for algo in ['superkmeans', 'faiss', 'scikit']:
        if algo in algo_data:
            df_algo = algo_data[algo]
            k_values = df_algo['n_clusters'].values
            times = df_algo['construction_time_ms'].values

            # Plot with large markers
            ax.scatter(k_values, times, s=120,
                      color=COLORS.get(algo, '#95a5a6'),
                      marker=markers.get(algo, 'o'),
                      label=labels.get(algo, algo),
                      edgecolors='black', linewidth=1.5,
                      alpha=0.85, zorder=3)

            # Add connecting line
            ax.plot(k_values, times, '-',
                   color=COLORS.get(algo, '#95a5a6'),
                   linewidth=2, alpha=0.5, zorder=2)

    # Calculate and annotate average speedup for each algorithm
    if 'superkmeans' in algo_data:
        skm_data = algo_data['superkmeans'].set_index('n_clusters')

        for algo, label in [('faiss', 'FAISS'), ('scikit', 'scikit')]:
            if algo in algo_data:
                speedups = []
                for _, row in algo_data[algo].iterrows():
                    k = row['n_clusters']
                    if k in skm_data.index:
                        speedup = row['construction_time_ms'] / skm_data.loc[k, 'construction_time_ms']
                        speedups.append(speedup)

                if speedups:
                    avg_speedup = np.mean(speedups)
                    # Find position for annotation
                    mid_k = algo_data[algo]['n_clusters'].iloc[len(algo_data[algo])//2]
                    mid_time = algo_data[algo]['construction_time_ms'].iloc[len(algo_data[algo])//2]

                    ax.annotate(f'{avg_speedup:.1f}x',
                               xy=(mid_k, mid_time),
                               xytext=(20, 20), textcoords='offset points',
                               fontsize=13, fontweight='bold',
                               color=COLORS.get(algo, '#95a5a6'),
                               bbox=dict(boxstyle='round,pad=0.5',
                                        facecolor='white',
                                        edgecolor=COLORS.get(algo, '#95a5a6'),
                                        linewidth=2, alpha=0.9))

    ax.set_xlabel('Number of Clusters (k)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Construction Time (ms)', fontweight='bold', fontsize=14)
    ax.set_title(f'K-Means Scaling Analysis - {dataset_name}\n' +
                 f'n={df_dataset.iloc[0]["data_size"]:,}, d={df_dataset.iloc[0]["dimensionality"]}',
                 fontweight='bold', fontsize=15, pad=15)

    # Log scale on both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Format tick labels
    ax.set_xticks(sorted(df_dataset['n_clusters'].unique()))
    ax.set_xticklabels([f'{k:,}' for k in sorted(df_dataset['n_clusters'].unique())])

    ax.legend(loc='upper left', framealpha=0.95, edgecolor='black',
             fancybox=False, fontsize=12)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'{dataset_name}_varying_k_scaling.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    # Default dataset name
    DEFAULT_DATASET = 'openai'

    # Parse arguments
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET

    # Determine CSV path
    script_dir = Path(__file__).parent
    benchmarks_dir = script_dir.parent

    if len(sys.argv) > 2:
        csv_path = Path(sys.argv[2])
    else:
        # Try common locations
        possible_locations = [
            benchmarks_dir / 'results' / 'intel' / 'varying_k.csv',
            benchmarks_dir / 'results' / 'default' / 'varying_k.csv',
        ]

        csv_path = None
        for path in possible_locations:
            if path.exists():
                csv_path = path
                break

        if csv_path is None:
            print(f"Error: CSV file not found in:")
            for path in possible_locations:
                print(f"  - {path}")
            print("\nSpecify path manually:")
            print(f"  python {sys.argv[0]} {dataset_name} /path/to/varying_k.csv")
            sys.exit(1)

    # Create output directory
    output_dir = script_dir / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Read CSV
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Filter by dataset
    df_dataset = df[df['dataset'] == dataset_name]

    if len(df_dataset) == 0:
        print(f"Error: No data found for dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(sorted(df['dataset'].unique()))}")
        sys.exit(1)

    print(f"Found {len(df_dataset)} rows for dataset '{dataset_name}'")
    print(f"Algorithms: {', '.join(sorted(df_dataset['algorithm'].unique()))}")
    print(f"k values: {', '.join(str(k) for k in sorted(df_dataset['n_clusters'].unique()))}")

    # Create both plots
    print("\nCreating speedup bar chart...")
    plot_speedup_bars(df_dataset, dataset_name, output_dir)

    print("\nCreating scaling plot...")
    plot_time_scaling(df_dataset, dataset_name, output_dir)

    print(f"\nPlots saved to: {output_dir}")
    print(f"\nUsage: python {Path(__file__).name} [dataset_name] [csv_path]")
    print(f"Default dataset: {DEFAULT_DATASET}")


if __name__ == '__main__':
    main()
