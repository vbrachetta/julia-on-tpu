#!/bin/bash

# julia_repl.sh - Start interactive Julia REPL in Docker container with TPU access

echo "Starting Julia REPL in Docker container..."

sudo docker run --rm -it --privileged \
  -v /dev:/dev \
  -v $(pwd):/workspace \
  --net=host \
  julia-tpu-image \
  julia
