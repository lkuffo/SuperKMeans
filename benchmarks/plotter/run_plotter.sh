#!/bin/bash
# Helper script to run the plotting script with the correct Python interpreter

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the project root directory (two levels up)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Path to the Python interpreter in the venv
PYTHON_VENV="$PROJECT_ROOT/venv/bin/python"

# Check if venv Python exists
if [ ! -f "$PYTHON_VENV" ]; then
    echo "Error: Python venv not found at $PYTHON_VENV"
    echo "Please create a virtual environment first:"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r benchmarks/plotter/requirements.txt"
    exit 1
fi

# Run the plotting script
echo "Running plotter script..."
cd "$SCRIPT_DIR"
"$PYTHON_VENV" plot_sweet_pruning_spot.py "$@"
