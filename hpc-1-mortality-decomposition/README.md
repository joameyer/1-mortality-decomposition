# HPC Chapter 1 Bundle

This folder is a self-contained upload bundle for running the Chapter 1
mortality preprocessing pipeline on the HPC cluster against the full ASIC data
that already exists in `hpc-icu-data-platform`.

It includes:
- the full `chapter1_mortality_decomposition` source package
- the Chapter 1 feature-set config
- a cluster-ready run config pointing at `hpc-icu-data-platform`
- a Python launcher that works without installing this repo as a package
- a shell wrapper and Slurm submission template
- the Chapter 1 preprocessing runbook notebook
- the observation-process / missingness visualization notebook

It does not include full ASIC data.

## Expected Cluster Layout

Upload this folder under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
```

The bundle assumes the upstream standardized ASIC artifacts live in the sibling
project:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-icu-data-platform/artifacts/
```

At runtime the loader now checks both common artifact folder names:
- `asic_harmonized`
- `asic_harmonized_full`

That should contain at least:
- `static/harmonized.csv`
- `dynamic/harmonized.csv`
- `blocked/asic_8h_block_index.csv`
- `blocked/asic_8h_blocked_dynamic_features.csv`
- `blocked/asic_8h_stay_block_counts.csv`
- `qc/mech_vent_ge_24h_stay_level.csv`
- `qc/mech_vent_ge_24h_episode_level.csv`

## Quick Start

### 1. Upload

Upload the whole folder as:

```bash
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
```

### 2. Install dependencies

From inside the uploaded folder:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already use a shared cluster environment, you can skip the local venv and
activate your existing environment instead.

### 3. Run preprocessing

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_preprocessing.sh
```

By default this resolves the sibling upstream artifact directory relative to the
bundle location and will use whichever of these exists:

```text
../hpc-icu-data-platform/artifacts/asic_harmonized
../hpc-icu-data-platform/artifacts/asic_harmonized_full
```

and writes Chapter 1 outputs to:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1
```

## Main Config

The default config file is:

`config/ch1_run_config.json`

The main defaults are:
- `input_dir`: `../hpc-icu-data-platform/artifacts/asic_harmonized`
- `output_dir`: `artifacts/chapter1`
- `input_format`: `csv`
- `output_format`: `csv`
- `feature_set_config_path`: `config/ch1_feature_sets.json`

If your cluster layout differs, edit `config/ch1_run_config.json` before
running or pass `INPUT_DIR` explicitly at submission time.

## Main Commands

### Direct Python launcher

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python run_chapter1_preprocessing.py --run-config config/ch1_run_config.json
```

If you want to bypass config path resolution entirely, provide the upstream
artifact directory explicitly:

```bash
python run_chapter1_preprocessing.py \
  --run-config config/ch1_run_config.json \
  --input-dir /rwthfs/rz/cluster/home/am861154/projects/hpc-icu-data-platform/artifacts/asic_harmonized_full
```

### Shell wrapper

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
./scripts/run_chapter1_preprocessing.sh --run-config config/ch1_run_config.json
```

### Slurm template

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch \
  --export=ALL,PROJECT_DIR=$PWD,RUN_CONFIG=$PWD/config/ch1_run_config.json \
  slurm/submit_chapter1_preprocessing.slurm
```

If needed, override the upstream artifact directory explicitly:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=$PWD,RUN_CONFIG=$PWD/config/ch1_run_config.json,INPUT_DIR=/rwthfs/rz/cluster/home/am861154/projects/hpc-icu-data-platform/artifacts/asic_harmonized_full \
  slurm/submit_chapter1_preprocessing.slurm
```

## Notebooks

The bundle includes:
- `notebooks/ch1_preprocessing_runbook.ipynb`
- `notebooks/ch1_observation_process_visualization.ipynb`

Both notebooks read the written `artifacts/chapter1` outputs. They are mainly
for inspection and visualization after the preprocessing job finishes.

Note:
- batch preprocessing does not require Jupyter
- opening or executing the notebooks does require a notebook-capable Python
  environment with IPython/Jupyter support

## Outputs

After a successful run, the main outputs are written under:

```text
artifacts/chapter1/
```

including:
- `cohort/`
- `instances/`
- `labels/`
- `splits/`
- `feature_sets/`
- `carry_forward/`
- `model_ready/`
- `observation_process/`

## Notes

- The launcher inserts `src/` into `sys.path`, so editable installation is not
  required.
- The preprocessing bundle is intentionally separate from `hpc-icu-data-platform`.
- This bundle is meant to consume already harmonized and blocked ASIC artifacts,
  not raw ASIC source files.
