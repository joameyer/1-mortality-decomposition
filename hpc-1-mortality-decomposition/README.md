# HPC Chapter 1 Bundle

This folder is a self-contained upload bundle for running the Chapter 1
mortality preprocessing pipeline, the ASIC logistic-regression baseline, the
ASIC XGBoost baseline, the baseline evaluation package, and the narrow ASIC 16h
temporal aggregation preview on the HPC cluster against the full ASIC data that
already exists in `hpc-icu-data-platform`.

It includes:
- the full `chapter1_mortality_decomposition` source package
- the Chapter 1 feature-set config
- a cluster-ready run config pointing at `hpc-icu-data-platform`
- a Python launcher that works without installing this repo as a package
- a shell wrapper and Slurm submission template
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  logistic-regression baseline
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  XGBoost baseline
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  baseline evaluation package
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  16h temporal aggregation preview
- the Chapter 1 preprocessing runbook notebook
- the observation-process / missingness visualization notebook
- the ASIC baseline evaluation review notebook

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

Optional editable install:

```bash
pip install -e .
```

That also registers these CLI commands inside the active environment:

- `chapter1-preprocess`
- `chapter1-logistic-baseline`
- `chapter1-xgboost-baseline`
- `chapter1-evaluate-baselines`
- `chapter1-temporal-preview`

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

### 4. Run the ASIC logistic-regression baseline

After preprocessing finishes, the logistic-regression baseline consumes:

```text
artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv
artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv
```

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_logistic_baseline.sh
```

By default this writes baseline outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/baselines/asic/primary_medians/logistic_regression
```

### 5. Run the ASIC XGBoost baseline

After preprocessing finishes, the XGBoost baseline consumes:

```text
artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv
artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv
```

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_xgboost_baseline.sh
```

By default this writes baseline outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/baselines/asic/primary_medians/xgboost
```

### 6. Run baseline evaluation

After both baseline model runs finish, the evaluation package reads the saved
prediction artifacts and writes metrics, figures, and a short interpretation
note.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_evaluate_baselines.sh
```

By default this writes evaluation outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/baselines/primary_medians
```

### 7. Run the ASIC 16h temporal aggregation preview

Run this only after the frozen 8h preprocessing and 8h baseline evaluation
artifacts already exist in the bundle, because the preview reuses the frozen
stay-level split assignments and compares back to the saved 8h evaluation.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_temporal_preview.sh
```

By default this writes the separate preview package under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/temporal_preview/asic/aggregation_16h
```

and writes the compact comparison notebook to:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/notebooks/ch1_asic_temporal_aggregation_preview_16h.ipynb
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

### Direct Python launcher for logistic baseline

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python run_chapter1_logistic_baseline.py
```

If needed, override the baseline inputs or restrict to selected horizons:

```bash
python run_chapter1_logistic_baseline.py \
  --input-dataset artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv \
  --feature-set-definition artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv \
  --output-dir artifacts/chapter1/baselines/asic/primary_medians/logistic_regression \
  --horizons 24 48
```

### Direct Python launcher for XGBoost baseline

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python run_chapter1_xgboost_baseline.py
```

If needed, override the baseline inputs or restrict to selected horizons:

```bash
python run_chapter1_xgboost_baseline.py \
  --input-dataset artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv \
  --feature-set-definition artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv \
  --output-dir artifacts/chapter1/baselines/asic/primary_medians/xgboost \
  --horizons 24 48
```

### Direct Python launcher for baseline evaluation

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python run_chapter1_evaluate_baselines.py
```

If needed, override the evaluation inputs or restrict models or horizons:

```bash
python run_chapter1_evaluate_baselines.py \
  --input-root artifacts/chapter1/baselines/asic/primary_medians \
  --output-dir artifacts/chapter1/evaluation/asic/baselines/primary_medians \
  --models logistic_regression xgboost \
  --horizons 24 48 \
  --primary-horizon 24
```

### Direct Python launcher for the 16h temporal aggregation preview

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python run_chapter1_temporal_preview.py
```

If needed, override the standardized ASIC input path, the frozen 8h artifact
roots, or the selected horizons:

```bash
python run_chapter1_temporal_preview.py \
  --run-config config/ch1_run_config.json \
  --input-dir /rwthfs/rz/cluster/home/am861154/projects/hpc-icu-data-platform/artifacts/asic_harmonized_full \
  --output-root artifacts/chapter1/temporal_preview/asic/aggregation_16h \
  --frozen-chapter1-dir artifacts/chapter1 \
  --eight-hour-evaluation-root artifacts/chapter1/evaluation/asic/baselines/primary_medians \
  --notebook-path notebooks/ch1_asic_temporal_aggregation_preview_16h.ipynb \
  --block-hours 16 \
  --horizons 8 16 24 48 72
```

### Shell wrapper

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
./scripts/run_chapter1_preprocessing.sh --run-config config/ch1_run_config.json
```

### Shell wrapper for logistic baseline

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
./scripts/run_chapter1_logistic_baseline.sh
```

### Shell wrapper for XGBoost baseline

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
./scripts/run_chapter1_xgboost_baseline.sh
```

### Shell wrapper for baseline evaluation

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
./scripts/run_chapter1_evaluate_baselines.sh
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

### Slurm template for logistic baseline

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch \
  --export=ALL,PROJECT_DIR=$PWD \
  slurm/submit_chapter1_logistic_baseline.slurm
```

If needed, override the horizons:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=$PWD,HORIZONS="24 48" \
  slurm/submit_chapter1_logistic_baseline.slurm
```

### Slurm template for XGBoost baseline

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch \
  --export=ALL,PROJECT_DIR=$PWD \
  slurm/submit_chapter1_xgboost_baseline.slurm
```

If needed, override the horizons:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=$PWD,HORIZONS="24 48" \
  slurm/submit_chapter1_xgboost_baseline.slurm
```

### Slurm template for baseline evaluation

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch \
  --export=ALL,PROJECT_DIR=$PWD \
  slurm/submit_chapter1_evaluate_baselines.slurm
```

If needed, restrict the models or horizons:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=$PWD,MODELS="logistic_regression xgboost",HORIZONS="24 48",PRIMARY_HORIZON=24 \
  slurm/submit_chapter1_evaluate_baselines.slurm
```

## Notebooks

The bundle includes:
- `notebooks/ch1_preprocessing_runbook.ipynb`
- `notebooks/ch1_observation_process_visualization.ipynb`
- `notebooks/ch1_asic_baseline_evaluation_review.ipynb`

These notebooks read the written `artifacts/chapter1` outputs. They are mainly
for inspection and visualization after preprocessing, baseline training, and
evaluation jobs finish.

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
- `baselines/asic/primary_medians/logistic_regression/`
- `baselines/asic/primary_medians/xgboost/`
- `evaluation/asic/baselines/primary_medians/`

The logistic baseline writes one subdirectory per horizon, each containing:
- `predictions.csv`
- `metrics.csv`
- `metadata.json`
- `selected_feature_columns.json`
- `preprocessing.pkl`
- `logistic_regression_model.pkl`
- `pipeline.pkl`

and the root baseline directory also contains:
- `horizon_run_summary.csv`
- `run_manifest.json`

The XGBoost baseline writes the same per-horizon structure, but with:
- `xgboost_model.pkl`

The evaluation package writes:
- `combined_metrics.csv`
- `reporting_split_summary.csv`
- `combined_risk_binned_summary.csv`
- `interpretation_note.md`
- per-model horizon comparison figures
- per-model reliability and mortality-vs-risk plots by horizon
- primary-horizon site sanity-check outputs
- `run_manifest.json`

## Notes

- The launcher inserts `src/` into `sys.path`, so editable installation is not
  required.
- The preprocessing bundle is intentionally separate from `hpc-icu-data-platform`.
- This bundle is meant to consume already harmonized and blocked ASIC artifacts,
  not raw ASIC source files.
