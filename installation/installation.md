# Julia installation on Google Cloud TPU VM

Installation attempts and outcomes on a Google Cloud TPU v4 VM running
Ubuntu 22.04.5 LTS. Five methods were tested.

---

## Method 1: juliaup (not working)

```bash
curl -fsSL https://install.julialang.org | sh
. ~/.bashrc
julia  # version 1.12.5
```

Julia starts, but any attempt to use `Pkg` — including `Pkg.status()` and
`Pkg.add` — aborts with a `double free or corruption` or
`munmap_chunk(): invalid pointer` signal originating in `Curl_parsenetrc`
within Julia's bundled `libcurl`. The crash occurs because libcurl attempts
to read a `.netrc` file during registry cloning and package resolution.
No working workaround was found. Containerised methods are recommended.

Full dumps:
- [`dumps/julia_juliaup_status_crash.txt`](dumps/julia_juliaup_status_crash.txt)
- [`dumps/julia_juliaup_addReactant_crash.txt`](dumps/julia_juliaup_addReactant_crash.txt)

**Secondary issue: abort on REPL exit**

Exiting the REPL with `exit()` triggers a second crash:

```
free(): invalid pointer
... close_unit_1 at libgfortran/io/unit.c ...
```

This is cosmetic — the session completes correctly before the abort. Use
Ctrl-D instead of `exit()`. The crash originates in an incompatibility between
Julia's bundled libgfortran (gcc-14) and the system glibc on Ubuntu 22.04.
Full dump: [`dumps/julia_juliaup_exit_crash.txt`](dumps/julia_juliaup_exit_crash.txt).

---

## Method 2: snap (not working)

```bash
sudo snap install julia --classic
# julia 1.12.5 from The Julia Language (julialang✓) installed
```

The `--classic` flag is required because Julia's snap package uses classic
confinement, meaning it operates outside the standard snap security sandbox.

`Pkg.status()` produces the identical `double free or corruption` crash as
Method 1, with the same `Curl_parsenetrc` root cause. The libcurl path differs
(`/snap/julia/165/...` vs the juliaup path), confirming the issue is
reproducible across installation methods and is not specific to juliaup.

Attempting to add a package produces the same crash. Exiting the REPL also
triggers the same exit crash as Method 1.

Full dumps:
- [`dumps/julia_snap_status_crash.txt`](dumps/julia_snap_status_crash.txt)
- [`dumps/julia_snap_addReactant_crash.txt`](dumps/julia_snap_addReactant_crash.txt)
- [`dumps/julia_snap_exit_crash.txt`](dumps/julia_snap_exit_crash.txt)

```bash
sudo snap remove julia
```

---

## Method 3: Docker (working)

Julia runs inside a container built from the official `julia:1.12.5` image,
with packages installed and precompiled at image build time. See the
[`dockerfile`](dockerfile) in the current directory.

### Building the image

```bash
sudo docker build -t julia-tpu-image .
```

### Interactive REPL

```bash
./julia_repl_docker.sh
```

### Running a script

```bash
./run_julia_docker.sh 01_reactant_test.jl
```

---

## Method 4: Podman (working)

Podman is a daemonless, rootless alternative to Docker. The same
[`dockerfile`](dockerfile) is used. Two differences apply compared to Docker:
the base image name must be fully qualified, and the `-f` flag is required
since Podman is case-sensitive when resolving the filename. Tested with
Podman 3.4.4.

### Building the image

```bash
podman build -t julia-tpu-image -f dockerfile \
    --from docker.io/library/julia:1.12.5 .
```

Locally built images are tagged with the `localhost/` prefix by Podman, which
is expected behaviour.

### Interactive REPL

```bash
chmod +x julia_repl_podman.sh
./julia_repl_podman.sh
```

### Running a script

```bash
chmod +x run_julia_podman.sh
./run_julia_podman.sh 01_reactant_test.jl
```

---

## Method 5: Apptainer (working)

An alternative containerised method producing a fully functional Julia
environment on this VM. Julia runs inside a container built from the official
`julia:1.12.5` Docker image, with packages installed and precompiled at image
build time using Apptainer. The definition file is provided as
[`julia.def`](julia.def) in the current directory.

### Building the image

```bash
apptainer build julia.sif julia.def
```

The rootless OCI extraction step emits a large number of `warn rootless{...}
ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"`
messages. These arise because Apptainer's non-setuid build mode cannot set
extended attributes on shared library files extracted from the Docker layer.
They do not affect the build outcome and can be ignored. A successful build
ends with:

```
INFO:    Build complete: julia.sif
```

### Interactive REPL

```bash
chmod +x julia_repl_apptainer.sh
./julia_repl_apptainer.sh
```

### Running a script

```bash
chmod +x run_julia_apptainer.sh
./run_julia_apptainer.sh 01_reactant_test.jl
```

### REPL history

The REPL history file (`$JULIA_DEPOT_PATH/logs/repl_history.jl`) cannot be
written to the read-only `.sif`. Julia falls back gracefully with
`[ Info: Disabling history file for this session ]`. To persist history across
sessions, redirect it to a writable host path:

```bash
apptainer exec \
    --env LD_PRELOAD="" \
    --env JULIA_HISTORY="$HOME/.julia_history" \
    julia.sif julia
```

### Pkg write access

Any `Pkg` operation — including `using` statements that trigger manifest usage
logging — attempts to write `manifest_usage.toml.pid` to
`$JULIA_DEPOT_PATH/logs/`, which fails on the read-only `.sif`. The fix is to
prepend a writable host directory to `JULIA_DEPOT_PATH`. Julia searches the
depot path left to right for reads and writes to the first entry:

```bash
apptainer exec \
    --env LD_PRELOAD="" \
    --env JULIA_DEPOT_PATH="$HOME/.julia_apptainer:/opt/julia_depot" \
    --env JULIA_HISTORY="$HOME/.julia_history" \
    julia.sif julia
```

The directory `~/.julia_apptainer` is created automatically on first use. The
baked-in packages in `/opt/julia_depot` remain available for reading. Both
wrapper scripts `run_julia_apptainer.sh` and `julia_repl_apptainer.sh` apply
this fix automatically.

### Notes on `LD_PRELOAD`

The `--env LD_PRELOAD=""` flag is required to prevent the host's
`libtcmalloc.so.4` (set in `LD_PRELOAD` by the TPU VM environment) from
leaking into the container, where it does not exist. Without it, Julia still
starts but emits a harmless linker error on every invocation. Both wrapper
scripts apply this flag automatically.

### Test script

The [`test_julia.jl`](test_julia.jl) script verifies that the baked-in
packages load correctly:

```bash
./run_julia_apptainer.sh test_julia.jl
```

---

## Summary

| Method    | Julia version | Starts | Pkg works | Notes                                     |
|:----------|:--------------|:------:|:---------:|:------------------------------------------|
| juliaup   | 1.12.5        | yes    | no        | libcurl crash; no working workaround      |
| snap      | 1.12.5        | yes    | no        | Same libcurl crash; snap removed          |
| Docker    | 1.12.5        | yes    | yes       | Working                                   |
| Podman    | 1.12.5        | yes    | yes       | Working; daemonless, rootless             |
| Apptainer | 1.12.5        | yes    | yes       | Working; no root required                 |
