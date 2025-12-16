#!/usr/bin/env python3
"""
Plot sweet_pruning_spot benchmark results.

For each dataset, creates a plot showing:
- X-axis: combination of min_not_pruned_pct and max_not_pruned_pct
- Y-axis: construction_time_ms
- Different markers for different adjustment_factor_for_partial_d values
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_config(config_str):
    """Parse the config JSON string and extract pruning parameters."""
    try:
        # Handle escaped quotes in CSV
        config_str = config_str.replace('""', '"')
        config = json.loads(config_str)
        return {
            'min_not_pruned_pct': float(config.get('min_not_pruned_pct', 0)),
            'max_not_pruned_pct': float(config.get('max_not_pruned_pct', 0)),
            'adjustment_factor_for_partial_d': float(
                config.get('adjustment_factor_for_partial_d', 0)
            ),
        }
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing config: {config_str}")
        print(f"Error: {e}")
        return None


def create_label(min_pct, max_pct):
    """Create a readable label for the pruning parameter pair."""
    return f"[{min_pct:.3f}, {max_pct:.3f}]"


def plot_dataset(df_dataset, dataset_name, output_dir):
    """Create a plot for a single dataset."""
    # Parse config for each row
    parsed_configs = []
    for _, row in df_dataset.iterrows():
        config = parse_config(row['config'])
        if config:
            parsed_configs.append(
                {
                    'construction_time_ms': row['construction_time_ms'],
                    'final_objective': row['final_objective'],
                    'min_not_pruned_pct': config['min_not_pruned_pct'],
                    'max_not_pruned_pct': config['max_not_pruned_pct'],
                    'adjustment_factor': config['adjustment_factor_for_partial_d'],
                }
            )

    if not parsed_configs:
        print(f"No valid configs found for dataset: {dataset_name}")
        return

    df_plot = pd.DataFrame(parsed_configs)

    # Create label for pruning parameter pairs
    df_plot['pruning_pair'] = df_plot.apply(
        lambda row: create_label(row['min_not_pruned_pct'], row['max_not_pruned_pct']),
        axis=1,
    )

    # Get unique adjustment factors and assign markers
    adjustment_factors = sorted(df_plot['adjustment_factor'].unique())
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker_map = {af: markers[i % len(markers)] for i, af in enumerate(adjustment_factors)}

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot each adjustment factor with different marker
    for af in adjustment_factors:
        df_af = df_plot[df_plot['adjustment_factor'] == af]
        ax.plot(
            df_af['pruning_pair'],
            df_af['construction_time_ms'],
            marker=marker_map[af],
            linestyle='-',
            linewidth=2,
            markersize=10,
            label=f'Adj. Factor = {af:.2f}',
            alpha=0.8,
        )

    # Customize plot
    ax.set_xlabel('Pruning Parameters [min, max]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Construction Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Sweet Pruning Spot Analysis - {dataset_name}\n'
        f'n={df_dataset.iloc[0]["data_size"]}, d={df_dataset.iloc[0]["dimensionality"]}, '
        f'k={df_dataset.iloc[0]["n_clusters"]}',
        fontsize=14,
        fontweight='bold',
    )

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'{dataset_name}_sweet_pruning_spot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")

    plt.close()


def plot_all_datasets_combined(df, output_dir):
    """Create a combined plot showing all datasets in subplots."""
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)

    if n_datasets == 0:
        print("No datasets found in CSV")
        return

    # Determine grid layout
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_datasets > 1 else [axes]

    for idx, dataset_name in enumerate(sorted(datasets)):
        df_dataset = df[df['dataset'] == dataset_name]
        ax = axes[idx]

        # Parse config for each row
        parsed_configs = []
        for _, row in df_dataset.iterrows():
            config = parse_config(row['config'])
            if config:
                parsed_configs.append(
                    {
                        'construction_time_ms': row['construction_time_ms'],
                        'min_not_pruned_pct': config['min_not_pruned_pct'],
                        'max_not_pruned_pct': config['max_not_pruned_pct'],
                        'adjustment_factor': config['adjustment_factor_for_partial_d'],
                    }
                )

        if not parsed_configs:
            continue

        df_plot = pd.DataFrame(parsed_configs)
        df_plot['pruning_pair'] = df_plot.apply(
            lambda row: create_label(row['min_not_pruned_pct'], row['max_not_pruned_pct']),
            axis=1,
        )

        # Get unique adjustment factors
        adjustment_factors = sorted(df_plot['adjustment_factor'].unique())
        markers = ['o', 's', '^']
        marker_map = {
            af: markers[i % len(markers)] for i, af in enumerate(adjustment_factors)
        }

        # Plot each adjustment factor
        for af in adjustment_factors:
            df_af = df_plot[df_plot['adjustment_factor'] == af]
            ax.plot(
                df_af['pruning_pair'],
                df_af['construction_time_ms'],
                marker=marker_map[af],
                linestyle='-',
                linewidth=1.5,
                markersize=6,
                label=f'AF={af:.2f}',
                alpha=0.7,
            )

        ax.set_title(dataset_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Pruning Params', fontsize=9)
        ax.set_ylabel('Time (ms)', fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        'Sweet Pruning Spot Analysis - All Datasets', fontsize=16, fontweight='bold', y=1.00
    )
    plt.tight_layout()

    # Save combined plot
    output_path = output_dir / 'all_datasets_combined_sweet_pruning_spot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")

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
        benchmarks_dir / 'results' / 'intel' / 'sweet_pruning_spot.csv',
        benchmarks_dir / 'results' / 'default' / 'sweet_pruning_spot.csv',
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
        print(f"  python {sys.argv[0]} /path/to/sweet_pruning_spot.csv")
        print("\nPlease run the sweet_pruning_spot benchmark first.")
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

    # Create combined plot
    print("\nCreating combined plot...")
    plot_all_datasets_combined(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
