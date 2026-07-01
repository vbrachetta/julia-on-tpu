#!/bin/bash
# julia_repl_apptainer.sh - Start interactive Julia REPL in Apptainer container with TPU access.
#
# LD_PRELOAD is cleared to prevent the host's libtcmalloc.so.4 (set in
# LD_PRELOAD by the TPU VM environment) from leaking into the container.
# JULIA_HISTORY is redirected to a writable host path to persist REPL history
# across sessions, since the .sif image is read-only.
# The .sif image is located in julia_installation/.

SIF="$(dirname "$0")/julia.sif"

# Check if .sif image exists
if [ ! -f "$SIF" ]; then
    echo "Error: Apptainer image '$SIF' not found!"
    echo "Build it first with: apptainer build julia_installation/julia.sif julia_installation/julia.def"
    exit 1
fi

echo "Starting Julia REPL in Apptainer container..."
apptainer exec \
    --env LD_PRELOAD="" \
    --env JULIA_DEPOT_PATH="$HOME/.julia_apptainer:/opt/julia_depot" \
    --env JULIA_HISTORY="$HOME/.julia_history" \
    "$SIF" \
    julia
