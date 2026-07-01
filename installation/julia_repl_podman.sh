#!/bin/bash
# julia_repl_podman.sh - Start interactive Julia REPL in Podman container with TPU access
#
# Note: Podman requires the fully qualified image name. Locally built images
# are tagged with the localhost/ prefix, which is used here automatically.

echo "Starting Julia REPL in Podman container..."
podman run --rm -it --privileged \
  -v /dev:/dev \
  -v $(pwd):/workspace \
  --net=host \
  localhost/julia-tpu-image \
  julia
