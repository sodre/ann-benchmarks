# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ann-benchmarks is a benchmarking framework for Approximate Nearest Neighbor (ANN) search algorithms. It runs 50+ algorithms in isolated Docker containers against standardized datasets and produces recall-vs-throughput plots.

## Common Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run unit tests
pytest

# Build all Docker images (or a specific one)
python install.py
python install.py --algorithm hnswlib
python install.py --proc 4          # parallel builds

# Run benchmarks
python run.py --dataset glove-100-angular
python run.py --algorithm annoy --dataset random-xs-20-angular --local   # no Docker
python run.py --docker-tag ann-benchmarks-hnswlib --max-n-algorithms 3 --runs 2 --timeout 300
python run.py --batch  # batch query mode

# Plot results (use .local.png to preserve original ann-benchmarks .png files)
python plot.py --dataset glove-100-angular --output results/glove-100-angular.local.png --x-scale logit --y-scale log

# Generate website from results
python create_website.py --outputdir website/ --scatter --latex

# Export results to CSV
python data_export.py --out results.csv
```

## Code Formatting

Black and Ruff with `line-length = 120` (configured in `pyproject.toml`).

## Architecture

### Execution Pipeline

`run.py` → `ann_benchmarks/main.py` (argument parsing, worker orchestration) → `ann_benchmarks/runner.py` (algorithm instantiation, index building, query execution, result storage)

Each algorithm runs inside its own Docker container. The host mounts `data/` (read-only) and `results/` (read-write). Inside the container, `runner.py:run_from_cmdline()` is the entry point that deserializes arguments, builds the index, runs queries, and writes HDF5 results.

### Algorithm Structure

Every algorithm lives in `ann_benchmarks/algorithms/{name}/` with three files:

- **`module.py`** — Python class inheriting `BaseANN` (`ann_benchmarks/algorithms/base/module.py`). Must implement `fit(X)` and `query(v, n)`. Optional: `set_query_arguments()`, `batch_query()`, `get_batch_results()`, `get_memory_usage()`, `get_additional()`, `done()`.
- **`config.yml`** — Defines supported point types (`float`/`bit`), distance metrics, constructor args, and run groups with parameter combinations. Special variables: `@metric`, `@count`, `@dimension`.
- **`Dockerfile`** — Extends `FROM ann-benchmarks` base image (Ubuntu 22.04 + Python 3.10 + HDF5). Installs algorithm dependencies and validates the import.

### Key Modules

- **`definitions.py`** — Loads all `config.yml` files, generates `Definition` dataclass instances by expanding parameter combinations (cartesian product of `args`/`query_args`).
- **`datasets.py`** — Downloads/creates HDF5 datasets. Each has keys: `train`, `test`, `neighbors`, `distances` plus attrs (`distance`, `dimension`, `point_type`).
- **`results.py`** — Reads/writes benchmark results as HDF5 files under `results/{dataset}/{count}/`.
- **`distance.py`** — Distance metric implementations (euclidean, angular, jaccard, hamming).
- **`plotting/metrics.py`** — Metric definitions for plotting (recall, QPS, build time, index size).
- **`plotting/utils.py`** — Pareto frontier computation for result visualization.

### Config Variable Substitution

In `config.yml`, `base_args: ['@metric']` gets substituted at runtime. The `definitions.py:_substitute_variables()` function replaces `@metric`, `@count`, and `@dimension` with actual values from the dataset.

### Docker Execution Model

`runner.py:run_docker()` creates a container per algorithm with CPU pinning (`cpuset_cpus`), memory limits, and a timeout. Logs stream to the host via a daemon thread. Single CPU per algorithm is enforced — use `--parallelism N` in `run.py` to run N containers simultaneously.

## Datasets

Stored in `data/` as HDF5 files. Common test dataset: `random-xs-20-angular` (small, fast). Production datasets: `glove-100-angular`, `sift-128-euclidean`, `fashion-mnist-784-euclidean`. Sparse datasets (Jaccard): `random-s-jaccard`, `kosarak-jaccard`.

## Results

HDF5 files in `results/{dataset}/{count}/{algo_name}-{params}.hdf5` containing query times, returned neighbors, distances, and metadata (build time, index size, search time).

### Fixing Docker Result Permissions

Docker containers run as root, so result HDF5 files are owned by root and unreadable by the host user. Fix with:

```bash
docker run --rm -v "$(pwd)/results:/results" ubuntu:22.04 chmod -R a+rw /results
```
