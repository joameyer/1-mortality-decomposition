# HPC Chapter 1 Bundle

This folder is a self-contained upload bundle for running the Chapter 1
mortality preprocessing pipeline, the ASIC logistic-regression baseline, the
ASIC XGBoost baseline, the baseline evaluation package, the ASIC horizon-
dependence comparison packages, and the narrow ASIC 16h temporal aggregation
preview on the HPC cluster against the full ASIC data that already exists in
`hpc-icu-data-platform`.

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
  XGBoost recalibration package
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  baseline evaluation package
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  hard-case definition package
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  hard-case agreement sensitivity package
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  hard-case comparison package
- a shell wrapper and Slurm submission template for the ASIC hard-case
  comparison variable audit
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  horizon-dependence foundation package
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  horizon hard-case stability package
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  final horizon-comparison package
- a shell wrapper and Slurm submission template for the ASIC ICD-10
  disease-group validation package
- a shell wrapper and Slurm submission template for the ASIC SOFA feasibility
  audit
- a Python launcher, shell wrapper, and Slurm submission template for the ASIC
  16h temporal aggregation preview
- the Chapter 1 preprocessing runbook notebook
- the observation-process / missingness visualization notebook
- the baseline model-readiness check notebook
- the ASIC baseline evaluation review notebook
- the XGBoost recalibration review notebook
- the ASIC hard-case review notebook
- the ASIC hard-case comparison notebook
- the ASIC 16h temporal aggregation preview notebook

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
- `chapter1-xgboost-recalibration`
- `chapter1-evaluate-baselines`
- `chapter1-define-hard-cases`
- `chapter1-hard-case-agreement`
- `chapter1-asic-hard-case-comparison`
- `chapter1-asic-horizon-dependence-foundation`
- `chapter1-asic-horizon-hard-case-stability`
- `chapter1-asic-horizon-dependence-final`
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

### 6. Run the ASIC XGBoost recalibration package

After the XGBoost baseline finishes, the recalibration package reads the saved
XGBoost prediction artifacts and writes recalibrated prediction tables for each
horizon.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_xgboost_recalibration.sh
```

By default this writes recalibration outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/recalibration/asic/primary_medians/xgboost
```

### 7. Run baseline evaluation

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

### 8. Run the ASIC hard-case definition package

After the logistic-regression baseline finishes, the hard-case package reads the
saved logistic prediction artifacts and writes the stay-level hard-case flags
plus the horizon summary table.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_define_hard_cases.sh
```

By default this writes hard-case outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression
```

### 9. Run the ASIC hard-case agreement sensitivity package

After the logistic baseline and XGBoost recalibration finish, the agreement
package reads the saved logistic and recalibrated XGBoost prediction artifacts
and writes the fatal-stay agreement table plus the horizon-level overlap
summary.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_hard_case_agreement.sh
```

By default this writes agreement outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_platt
```

The first-pass review notebook for these artifacts lives at:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/notebooks/ch1_asic_hard_case_review.ipynb
```

### 10. Run the ASIC 16h temporal aggregation preview

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

### 11. Run the ASIC horizon-dependence foundation package

After the hard-case definition job finishes, this package validates the saved
stay-level hard-case artifacts across the five frozen horizons and writes the
foundation summary tables and note.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_asic_horizon_dependence_foundation.sh
```

By default this writes outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/horizon_dependence/foundation
```

### 12. Run the ASIC hard-case stability package

After the hard-case definition job finishes, this package computes pairwise
hard-case overlap, directional overlap, and persistence across the frozen
horizons.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_asic_horizon_hard_case_stability.sh
```

By default this writes outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/horizon_dependence/overlap
```

### 13. Run the final ASIC horizon-comparison package

After the foundation and hard-case stability jobs finish, this package writes
the five-panel mortality-vs-risk comparison figure plus the short interpretation
memo and final summary note.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_asic_horizon_dependence_final.sh
```

By default this writes outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/horizon_dependence/final
```

### 14. Submit the full horizon-dependence chain in one command

If you always run the three horizon packages together, use the convenience
wrapper below. It submits the foundation job, then the overlap job with an
`afterok` dependency, then the final figure/memo job with an `afterok`
dependency on the overlap job.

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
bash run_asic_horizon_dependence_pipeline.sh
```

This keeps the jobs separate at the scheduler level while giving you a single
entrypoint.

### 15. Run the ASIC ICD-10 disease-group validation package

This job inspects the upstream ASIC static `icd10_codes` field on the full
cluster artifact, applies the frozen six-group hierarchy, and writes reviewable
counts, ambiguity summaries, sample rows, and a short validation memo.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_asic_icd10_disease_group_validation.sh
```

By default this writes outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/icd10_disease_group_validation
```

### 16. Run the ASIC hard-case comparison package

After preprocessing, the logistic baseline, and hard-case definition finish,
this job builds the fatal-stay comparison package used for Issue 3.2 follow-up
review.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_asic_hard_case_comparison.sh
```

By default this writes outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison
```

### 17. Run the hard-case comparison variable audit

After preprocessing, the logistic baseline, and hard-case definition finish,
this utility audit writes a compact table and memo about variable availability
for the 24h comparison package.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_asic_hard_case_comparison_variable_audit.sh
```

By default this writes outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit
```

### 18. Run the ASIC SOFA feasibility audit

After preprocessing, the logistic baseline, and hard-case definition finish,
this utility audit writes the SOFA component feasibility table and memo for the
same follow-up review path.

The simplest path is:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch run_sofa_feasibility.sh
```

By default this writes outputs under:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit
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

### Direct Python launchers for additional Chapter 1 analysis packages

Run any of the follow-on analysis packages directly with:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python run_chapter1_xgboost_recalibration.py
python run_chapter1_define_hard_cases.py
python run_chapter1_hard_case_agreement.py
python run_chapter1_asic_hard_case_comparison.py
python run_chapter1_asic_horizon_dependence_foundation.py
python run_chapter1_asic_horizon_hard_case_stability.py
python run_chapter1_asic_horizon_dependence_final.py
```

The standalone utility audits use:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python scripts/ch1_asic_hard_case_comparison_variable_audit.py
python scripts/ch1_sofa_feasibility_audit.py
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

### Direct Python launcher for ASIC ICD-10 disease-group validation

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
python scripts/ch1_asic_icd10_disease_group_inspection.py
```

If needed, override the upstream artifact root explicitly:

```bash
ASIC_INPUT_ROOT=/rwthfs/rz/cluster/home/am861154/projects/hpc-icu-data-platform/artifacts/asic_harmonized_full \
python scripts/ch1_asic_icd10_disease_group_inspection.py
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

### Additional shell wrappers

The bundle also includes these wrapper entrypoints:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
./scripts/run_chapter1_xgboost_recalibration.sh
./scripts/run_chapter1_define_hard_cases.sh
./scripts/run_chapter1_hard_case_agreement.sh
./scripts/run_chapter1_asic_hard_case_comparison.sh
./scripts/run_chapter1_asic_horizon_dependence_foundation.sh
./scripts/run_chapter1_asic_horizon_hard_case_stability.sh
./scripts/run_chapter1_asic_horizon_dependence_final.sh
./scripts/run_chapter1_temporal_preview.sh
./scripts/run_chapter1_asic_hard_case_comparison_variable_audit.sh
./scripts/run_chapter1_sofa_feasibility.sh
```

### Shell wrapper for ASIC ICD-10 disease-group validation

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
./scripts/run_chapter1_asic_icd10_disease_group_validation.sh
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

### Additional Slurm templates

The bundle also includes these ready-to-submit templates:

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch --export=ALL,PROJECT_DIR=$PWD slurm/submit_chapter1_xgboost_recalibration.slurm
sbatch --export=ALL,PROJECT_DIR=$PWD slurm/submit_chapter1_define_hard_cases.slurm
sbatch --export=ALL,PROJECT_DIR=$PWD slurm/submit_chapter1_hard_case_agreement.slurm
sbatch --export=ALL,PROJECT_DIR=$PWD slurm/submit_chapter1_temporal_preview.slurm
sbatch --export=ALL,PROJECT_DIR=$PWD slurm/submit_chapter1_asic_hard_case_comparison.slurm
sbatch --export=ALL,PROJECT_DIR=$PWD slurm/submit_chapter1_asic_hard_case_comparison_variable_audit.slurm
sbatch --export=ALL,PROJECT_DIR=$PWD slurm/submit_chapter1_sofa_feasibility.slurm
```

### Slurm template for ASIC ICD-10 disease-group validation

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch \
  --export=ALL,PROJECT_DIR=$PWD \
  slurm/submit_chapter1_asic_icd10_disease_group_validation.slurm
```

If needed, override the upstream artifact directory explicitly:

```bash
sbatch \
  --export=ALL,PROJECT_DIR=$PWD,ASIC_INPUT_ROOT=/rwthfs/rz/cluster/home/am861154/projects/hpc-icu-data-platform/artifacts/asic_harmonized_full \
  slurm/submit_chapter1_asic_icd10_disease_group_validation.slurm
```

### Slurm template for ASIC horizon-dependence foundation

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch \
  --export=ALL,PROJECT_DIR=$PWD \
  slurm/submit_chapter1_asic_horizon_dependence_foundation.slurm
```

### Slurm template for ASIC hard-case stability

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch \
  --export=ALL,PROJECT_DIR=$PWD \
  slurm/submit_chapter1_asic_horizon_hard_case_stability.slurm
```

### Slurm template for final ASIC horizon comparison

```bash
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
sbatch \
  --export=ALL,PROJECT_DIR=$PWD \
  slurm/submit_chapter1_asic_horizon_dependence_final.slurm
```

## Notebooks

The bundle includes:
- `notebooks/ch1_preprocessing_runbook.ipynb`
- `notebooks/ch1_observation_process_visualization.ipynb`
- `notebooks/ch1_baseline_model_readiness_check.ipynb`
- `notebooks/ch1_asic_baseline_evaluation_review.ipynb`
- `notebooks/ch1_xgboost_recalibration_review.ipynb`
- `notebooks/ch1_asic_hard_case_review.ipynb`
- `notebooks/ch1_asic_hard_case_comparison.ipynb`
- `notebooks/ch1_asic_temporal_aggregation_preview_16h.ipynb`

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
- `evaluation/asic/hard_cases/primary_medians/logistic_regression/`
- `evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison/`
- `evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_hard_case_comparison_variable_audit/`
- `evaluation/asic/hard_cases/primary_medians/logistic_regression/asic_sofa_feasibility_audit/`
- `evaluation/asic/horizon_dependence/foundation/`
- `evaluation/asic/horizon_dependence/overlap/`
- `evaluation/asic/horizon_dependence/final/`

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
