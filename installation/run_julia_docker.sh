#!/bin/bash
#
# run_julia.sh - Run a Julia script inside the julia-tpu-image Docker container.
#
# Usage:   ./run_julia.sh <script.jl>
# Example: ./run_julia.sh 01_reactant_test.jl
#
# The container is granted privileged access and /dev is mounted so that
# the TPU device is visible to the XLA runtime. --net=host is required for
# the TPU runtime to communicate correctly on TRC VMs.
# The --rm flag ensures the container is removed automatically on exit,
# keeping the environment tidy.

# Check if a file argument was provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_julia.sh <julia_file.jl>"
    echo "Example: ./run_julia.sh 01_reactant_test.jl"
    exit 1
fi

JULIA_FILE="$1"

# Check if file exists
if [ ! -f "$JULIA_FILE" ]; then
    echo "Error: File '$JULIA_FILE' not found!"
    exit 1
fi

echo "Running $JULIA_FILE in Docker container..."
sudo docker run --rm -it --privileged \
  -v /dev:/dev \
  -v $(pwd):/workspace \
  --net=host \
  julia-tpu-image \
  julia /workspace/"$JULIA_FILE"
