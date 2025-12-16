#!/usr/bin/env python3
"""
Plot pruning_bailout_analysis benchmark results.

For each dataset, creates a bar chart comparing:
- with_pruning mode (use_blas_only=false)
- blas_only mode (use_blas_only=true)

Shows construction time and speedup ratios.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_config(config_str):
    """Parse the config JSON string and extract relevant parameters."""
    try:
        # Handle escaped quotes in CSV
        config_str = config_str.replace('""', '"')
        config = json.loads(config_str)
        return {
            'use_blas_only': config.get('use_blas_only', 'false') == 'true',
            'mode': config.get('mode', 'unknown').strip('"'),
        }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing config: {config_str}")
        print(f"Error: {e}")
        return None


def plot_dataset(df_dataset, dataset_name, output_dir):
    """Create a bar chart for a single dataset."""
    # Parse config for each row
    parsed_data = []
    for _, row in df_dataset.iterrows():
        config = parse_config(row['config'])
        if config:
            parsed_data.append(
                {
                    'mode': config['mode'],
                    'use_blas_only': config['use_blas_only'],
                    'construction_time_ms': row['construction_time_ms'],
                    'final_objective': row['final_objective'],
                }
            )

    if not parsed_data:
        print(f"No valid configs found for dataset: {dataset_name}")
        return

    df_plot = pd.DataFrame(parsed_data)

    # Get times for both modes
    with_pruning_time = df_plot[df_plot['mode'] == 'with_pruning']['construction_time_ms'].values
    blas_only_time = df_plot[df_plot['mode'] == 'blas_only']['construction_time_ms'].values

    if len(with_pruning_time) == 0 or len(blas_only_time) == 0:
        print(f"Missing data for dataset: {dataset_name}")
        return

    with_pruning_time = with_pruning_time[0]
    blas_only_time = blas_only_time[0]

    # Calculate speedup
    speedup = blas_only_time / with_pruning_time  if with_pruning_time > 0 else 0

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Bar chart comparing times
    modes = ['With Pruning', 'BLAS Only']
    times = [with_pruning_time, blas_only_time]
    colors = ['#2ecc71', '#e74c3c']  # Green for pruning, red for BLAS only

    bars = ax1.bar(modes, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{time:.1f} ms',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
        )

    ax1.set_ylabel('Construction Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Training Time Comparison - {dataset_name}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Subplot 2: Speedup bar
    speedup_colors = ['#3498db']  # Blue for speedup
    speedup_bar = ax2.bar(['Speedup'], [speedup], color=speedup_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value label
    ax2.text(
        speedup_bar[0].get_x() + speedup_bar[0].get_width() / 2.0,
        speedup,
        f'{speedup:.2f}x',
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold',
    )

    # Add horizontal line at y=1 (no speedup)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (1x)')

    ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax2.set_title(f'Speedup (BLAS Only / With Pruning)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)

    # Add dataset info as subtitle
    fig.suptitle(
        f'n={df_dataset.iloc[0]["data_size"]:,}, d={df_dataset.iloc[0]["dimensionality"]}, '
        f'k={df_dataset.iloc[0]["n_clusters"]}',
        fontsize=11,
        y=0.98,
    )

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'{dataset_name}_pruning_bailout_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")

    plt.close()


def plot_all_datasets_combined(df, output_dir):
    """Create a combined plot showing all datasets with grouped bar charts."""
    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)

    if n_datasets == 0:
        print("No datasets found in CSV")
        return

    # Collect data for all datasets
    all_data = []
    for dataset_name in datasets:
        df_dataset = df[df['dataset'] == dataset_name]

        # Parse config for each row
        for _, row in df_dataset.iterrows():
            config = parse_config(row['config'])
            if config:
                all_data.append(
                    {
                        'dataset': dataset_name,
                        'mode': config['mode'],
                        'construction_time_ms': row['construction_time_ms'],
                    }
                )

    if not all_data:
        print("No valid data found")
        return

    df_all = pd.DataFrame(all_data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(max(12, n_datasets * 1.5), 8))

    # Prepare data for grouped bars
    with_pruning_times = []
    blas_only_times = []
    speedups = []

    for dataset in datasets:
        df_ds = df_all[df_all['dataset'] == dataset]
        with_pruning = df_ds[df_ds['mode'] == 'with_pruning']['construction_time_ms'].values
        blas_only = df_ds[df_ds['mode'] == 'blas_only']['construction_time_ms'].values

        if len(with_pruning) > 0 and len(blas_only) > 0:
            with_pruning_times.append(with_pruning[0])
            blas_only_times.append(blas_only[0])
            speedups.append(blas_only[0] / with_pruning[0] if with_pruning[0] > 0 else 0)
        else:
            with_pruning_times.append(0)
            blas_only_times.append(0)
            speedups.append(0)

    # Set up bar positions
    x = np.arange(len(datasets))
    width = 0.35

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        with_pruning_times,
        width,
        label='With Pruning',
        color='#2ecc71',
        alpha=0.8,
        edgecolor='black',
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        blas_only_times,
        width,
        label='BLAS Only',
        color='#e74c3c',
        alpha=0.8,
        edgecolor='black',
        linewidth=1,
    )

    # Add speedup annotations above the bars
    for i, (dataset, speedup) in enumerate(zip(datasets, speedups)):
        y_max = max(with_pruning_times[i], blas_only_times[i])
        ax.text(
            i,
            y_max * 1.05,
            f'{speedup:.2f}x',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#3498db',
        )

    # Customize plot
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Construction Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Pruning Bailout Analysis - All Datasets\nWith Pruning vs BLAS Only',
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()

    # Save combined plot
    output_path = output_dir / 'all_datasets_combined_pruning_bailout_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")

    plt.close()

    # Create a second plot showing just speedups
    fig, ax = plt.subplots(figsize=(max(10, n_datasets * 1.2), 6))

    bars = ax.bar(
        datasets, speedups, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5
    )

    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (1x)')

    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{speedup:.2f}x',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Factor (BLAS Only / With Pruning)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Analysis - All Datasets', fontsize=14, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()

    # Save speedup plot
    output_path = output_dir / 'all_datasets_speedup_pruning_bailout_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved speedup plot: {output_path}")

    plt.close()


def main():
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'

    # Determine paths
    script_dir = Path(__file__).parent
    benchmarks_dir = script_dir.parent

    # Check for CSV in multiple possible locations
    possible_locations = [
        benchmarks_dir / 'results' / 'intel' / 'pruning_bailout_analysis.csv',
        benchmarks_dir / 'results' / 'default' / 'pruning_bailout_analysis.csv',
    ]

    csv_path = None
    for path in possible_locations:
        if path.exists():
            csv_path = path
            break

    # Allow command line argument to override
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])

    # Create output directory for plots
    output_dir = script_dir / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Check if CSV exists
    if csv_path is None or not csv_path.exists():
        print(f"Error: CSV file not found in any of these locations:")
        for path in possible_locations:
            print(f"  - {path}")
        print("\nYou can also specify a custom path:")
        print(f"  python {sys.argv[0]} /path/to/pruning_bailout_analysis.csv")
        print("\nPlease run the pruning_bailout_analysis benchmark first.")
        sys.exit(1)

    # Read CSV
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Get unique datasets
    datasets = df['dataset'].unique()
    print(f"Found {len(datasets)} dataset(s): {', '.join(datasets)}")

    # Create individual plots for each dataset
    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        df_dataset = df[df['dataset'] == dataset_name]
        plot_dataset(df_dataset, dataset_name, output_dir)

    # Create combined plots
    print("\nCreating combined plots...")
    plot_all_datasets_combined(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
