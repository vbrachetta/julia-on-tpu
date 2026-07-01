#!/bin/bash
#
# run_julia_apptainer.sh - Run a Julia script inside the julia.sif Apptainer container.
#
# Usage:   ./run_julia_apptainer.sh <script.jl>
# Example: ./run_julia_apptainer.sh 01_reactant_test.jl
#
# LD_PRELOAD is cleared to prevent the host's libtcmalloc.so.4 (set in
# LD_PRELOAD by the TPU VM environment) from leaking into the container.
# The .sif image is located in julia_installation/.

SIF="$(dirname "$0")/julia.sif"

# Check if a file argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_julia_apptainer.sh <julia_file.jl>"
    echo "Example: ./run_julia_apptainer.sh 01_reactant_test.jl"
    exit 1
fi

JULIA_FILE="$1"

# Check if file exists
if [ ! -f "$JULIA_FILE" ]; then
    echo "Error: File '$JULIA_FILE' not found!"
    exit 1
fi

# Check if .sif image exists
if [ ! -f "$SIF" ]; then
    echo "Error: Apptainer image '$SIF' not found!"
    echo "Build it first with: apptainer build julia_installation/julia.sif julia_installation/julia.def"
    exit 1
fi

echo "Running $JULIA_FILE in Apptainer container..."
apptainer exec \
    --env LD_PRELOAD="" \
    "$SIF" \
    julia "$JULIA_FILE"
