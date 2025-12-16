# Sweet Pruning Spot Plotter

This directory contains plotting scripts for visualizing the sweet_pruning_spot benchmark results.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda install pandas matplotlib seaborn
```

## Usage

1. First, run the sweet_pruning_spot benchmark to generate the CSV file:

```bash
cd /path/to/SuperKMeans/cmake-build-release
./benchmarks/sweet_pruning_spot_superkmeans.out
```

2. Run the plotting script:

**Option A: Using the helper script (recommended)**
```bash
cd /path/to/SuperKMeans/benchmarks/plotter
./run_plotter.sh
```

**Option B: Using Python directly**
```bash
cd /path/to/SuperKMeans/benchmarks/plotter
python plot_sweet_pruning_spot.py
# Or specify a custom CSV path:
python plot_sweet_pruning_spot.py /path/to/custom/sweet_pruning_spot.csv
```

**Option C: Using the venv Python**
```bash
cd /path/to/SuperKMeans/benchmarks/plotter
../../venv/bin/python plot_sweet_pruning_spot.py
```

The script will:
- Read the CSV file from `../results/intel/sweet_pruning_spot.csv` or `../results/default/sweet_pruning_spot.csv`
- Parse the config JSON to extract the three pruning parameters
- Generate individual plots for each dataset
- Generate a combined plot showing all datasets
- Save all plots to the `plots/` subdirectory

## Output

The script creates:
- **Individual plots**: One plot per dataset (e.g., `arxiv_sweet_pruning_spot.png`)
- **Combined plot**: All datasets in a grid layout (`all_datasets_combined_sweet_pruning_spot.png`)

Each plot shows:
- **X-axis**: Pruning parameter pairs [min_not_pruned_pct, max_not_pruned_pct]
- **Y-axis**: Construction time in milliseconds
- **Different markers**: Each adjustment_factor_for_partial_d value (0.10, 0.15, 0.20)
- **Lines**: Connect points with the same adjustment factor

## Plot Interpretation

- **Lower is better**: Lower construction times indicate more efficient parameter settings
- **Compare markers**: Different markers show how the adjustment factor affects performance
- **Compare x-positions**: Different pruning parameter pairs show the sweet spot range
- **Look for valleys**: The optimal combination is where construction time is minimized
