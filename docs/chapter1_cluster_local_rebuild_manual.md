# Chapter 1 Cluster-Local Rebuild Manual

## Purpose

This manual is the clean rebuild path for Chapter 1.

Use it when you want to:

1. delete the old local and cluster outputs
2. rerun the protected-data analysis from scratch on the cluster
3. stage only the approved local-review exports
4. import those exports into the local repo
5. regenerate the local review notebooks, reports, and presentation artifacts

This is the authoritative "how to reconstruct the project" guide.

## Core Principle

- Protected patient-level computation stays on the cluster.
- Local scientific review uses approved exports under `cluster-results/chapter1_true_results/`.
- Local `artifacts/chapter1/` is for synthetic or development-only output unless explicitly stated otherwise.

## Assumptions

This manual assumes:

- the local repo is the current repo root
- the HPC upload bundle is `hpc-1-mortality-decomposition/`
- the cluster project path is `/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition`
- the upstream protected ASIC artifacts already exist at `/rwthfs/rz/cluster/home/am861154/projects/hpc-icu-data-platform/artifacts/`
- you have SSH access to the cluster

## One-Time Variables

Set these locally before you start:

```bash
export LOCAL_REPO="/Users/joanameyer/repository/1-mortality-decomposition"
export LOCAL_HPC_BUNDLE="$LOCAL_REPO/hpc-1-mortality-decomposition"
export CLUSTER_HOST="am861154@login23-1.hpc.itc.rwth-aachen.de"
export CLUSTER_PROJECT_DIR="/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition"
```

Replace `<your-cluster-login-host>` with the actual SSH host you use.

## Phase 0: Clean Previous Outputs

Run this only if you really want a full clean rebuild.

### 0.1 Clean Local Outputs

```bash
cd "$LOCAL_REPO"
rm -rf artifacts/chapter1
rm -rf cluster-results/chapter1_true_results
rm -rf export-staging/chapter1_true_results
```

Optional: also remove regenerated local review outputs if you want them rebuilt too:

```bash
cd "$LOCAL_REPO"
rm -f notebooks/ch1_asic_descriptive_viability_review.ipynb
rm -f reports/ch1_asic_descriptive_viability_evidence_pack.md
rm -f reports/ch1_asic_descriptive_viability_memo_draft.md
```

### 0.2 Clean Cluster Outputs

```bash
ssh "$CLUSTER_HOST" <<'EOF'
cd /rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition
rm -rf artifacts/chapter1
rm -rf export-staging/chapter1_true_results
rm -f logs/*.out logs/*.err
mkdir -p logs
EOF
```

## Phase 1: Upload or Refresh the HPC Bundle

Upload the bundle from local to the cluster with `rsync`.

This command refreshes code, configs, scripts, notebooks, and Slurm files, but does not upload local artifacts, logs, or virtual environments.

```bash
rsync -av \
  --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'artifacts' \
  --exclude 'artifacts-old' \
  --exclude 'notebooks-old' \
  --exclude 'logs' \
  --exclude 'export-staging' \
  "$LOCAL_HPC_BUNDLE"/ \
  "${CLUSTER_HOST}:${CLUSTER_PROJECT_DIR}/"
```

## Phase 2: Prepare the Cluster Environment

```bash
ssh "$CLUSTER_HOST"
cd "$CLUSTER_PROJECT_DIR"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you already use an existing cluster environment, activate that instead.

## Phase 3: Run the Cluster Pipeline

General rule:

- wait for each dependency step to finish successfully before launching downstream steps
- use `squeue -u "$USER"` to monitor jobs
- inspect `logs/` if a job fails

Useful monitoring commands on the cluster:

```bash
cd "$CLUSTER_PROJECT_DIR"
squeue -u "$USER"
ls logs
tail -f logs/<job-log-file>.out
```

### 3.1 Preprocessing

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_preprocessing.sh
```

Wait until preprocessing finishes successfully.

### 3.2 Baseline Models

After preprocessing succeeds:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_logistic_baseline.sh
sbatch run_xgboost_baseline.sh
```

These can run in parallel.

Wait until both finish successfully.

### 3.3 Recalibration and Baseline Evaluation

After the XGBoost baseline finishes:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_xgboost_recalibration.sh
```

After both baseline models finish:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_evaluate_baselines.sh
```

### 3.4 Hard-Case Definition

After the logistic baseline finishes:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_define_hard_cases.sh
```

### 3.5 Hard-Case Comparison and Related Audits

After preprocessing, the logistic baseline, and hard-case definition finish:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_asic_hard_case_comparison.sh
sbatch run_asic_hard_case_comparison_variable_audit.sh
sbatch run_sofa_feasibility.sh
```

### 3.6 Hard-Case Agreement

After hard-case definition and XGBoost recalibration finish:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_hard_case_agreement.sh
```

### 3.7 Temporal Aggregation Preview

After preprocessing and baseline evaluation finish:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_temporal_preview.sh
```

### 3.8 Horizon Dependence

After hard-case definition finishes:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_asic_horizon_dependence_foundation.sh
sbatch run_asic_horizon_hard_case_stability.sh
```

After both of those finish:

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_asic_horizon_dependence_final.sh
```

Optional convenience wrapper:

```bash
cd "$CLUSTER_PROJECT_DIR"
bash run_asic_horizon_dependence_pipeline.sh
```

### 3.9 ICD-10 Disease-Group Validation

This is not needed for the main baseline/hard-case pipeline, but it is part of the Chapter 1 review stack.

```bash
cd "$CLUSTER_PROJECT_DIR"
sbatch run_asic_icd10_disease_group_validation.sh
```

## Phase 4: Stage Approved Local-Review Exports on the Cluster

After all needed cluster jobs finish successfully, stage the approved local-review bundles.

Run this on the cluster:

```bash
cd "$CLUSTER_PROJECT_DIR"
source .venv/bin/activate
python run_chapter1_stage_local_review_exports.py \
  --include-foundational-summaries \
  --include-baseline-predictions \
  --include-baseline-evaluation \
  --include-hard-case-definition \
  --include-xgboost-recalibration \
  --include-hard-case-agreement \
  --include-horizon-dependence \
  --include-variable-audit \
  --include-sofa-feasibility \
  --include-temporal-preview \
  --include-icd10-validation
```

Important note:
- If you are overwriting over an existing staged local-review bundles use '--overwrite'
- `notebooks/ch1_asic_hard_case_review.ipynb` also needs the hard-case definition package itself, so include `--include-hard-case-definition`
- `notebooks/ch1_risk_trajectory_shapes.ipynb` needs the raw baseline prediction exports, so include `--include-baseline-predictions`
- the default hard-case comparison aggregate bundle is staged automatically
- the staging helper mirrors only the approved local-review outputs
- restricted row-level tables are intentionally excluded from the default local mirror

The staged export root on the cluster is:

```text
/rwthfs/rz/cluster/home/am861154/projects/hpc-1-mortality-decomposition/export-staging/chapter1_true_results
```

## Phase 5: Copy the Staged Export Tree Back to Local

Run this locally:

```bash
mkdir -p "$LOCAL_REPO/export-staging/chapter1_true_results"

rsync -av \
  "${CLUSTER_HOST}:${CLUSTER_PROJECT_DIR}/export-staging/chapter1_true_results/" \
  "$LOCAL_REPO/export-staging/chapter1_true_results/"
```

## Phase 6: Import the Staged Exports into the Local Mirror

Run this locally:

```bash
cd "$LOCAL_REPO"
python run_chapter1_import_staged_exports.py
```

If you are re-importing over an existing local mirror, use:

```bash
cd "$LOCAL_REPO"
python run_chapter1_import_staged_exports.py --overwrite
```

After this step, the authoritative local mirror is:

```text
cluster-results/chapter1_true_results/
```

## Phase 7: Local Review and Regeneration

### 7.1 Local Python Environment

If needed, prepare the local environment:

```bash
cd "$LOCAL_REPO"
python -m pip install -e ".[dev]"
```

### 7.2 Regenerate the Viability Review Outputs

```bash
cd "$LOCAL_REPO"
PYTHONPATH=src python -m chapter1_mortality_decomposition.ch1_asic_descriptive_viability --repo-root .
```

This regenerates:

- `notebooks/ch1_asic_descriptive_viability_review.ipynb`
- `reports/ch1_asic_descriptive_viability_evidence_pack.md`
- `reports/ch1_asic_descriptive_viability_memo_draft.md`

### 7.3 Open the Main Local Review Notebooks

Run whichever of these you need locally:

```bash
cd "$LOCAL_REPO"
jupyter notebook notebooks/ch1_asic_baseline_evaluation_review.ipynb
jupyter notebook notebooks/ch1_xgboost_recalibration_review.ipynb
jupyter notebook notebooks/ch1_asic_hard_case_review.ipynb
jupyter notebook notebooks/ch1_asic_hard_case_comparison_local_review.ipynb
jupyter notebook notebooks/ch1_risk_trajectory_shapes.ipynb
```

For the imported temporal preview notebook:

```bash
cd "$LOCAL_REPO"
jupyter notebook cluster-results/chapter1_true_results/temporal_preview/asic/aggregation_16h/comparison/preview_review.ipynb
```

Important note:

- `notebooks/ch1_asic_hard_case_comparison.ipynb` is still the cluster-side or explicitly approved row-level notebook
- for local review, use `notebooks/ch1_asic_hard_case_comparison_local_review.ipynb`

### 7.4 Rebuild the Methods/Results Presentation

If your local environment has `python-pptx` installed:

```bash
cd "$LOCAL_REPO"
python scripts/build_ch1_methods_results_presentation.py
```

## Recommended Full Rebuild Order

If you want the shortest safe checklist, follow this exact order:

1. clean local outputs
2. clean cluster outputs
3. upload the HPC bundle with `rsync`
4. install or reactivate the cluster environment
5. run preprocessing
6. run logistic baseline and XGBoost baseline
7. run XGBoost recalibration
8. run baseline evaluation
9. run hard-case definition
10. run hard-case comparison, variable audit, and SOFA feasibility
11. run hard-case agreement
12. run temporal preview
13. run horizon foundation and horizon stability
14. run horizon final
15. run ICD-10 disease-group validation
16. stage approved local-review exports on the cluster
17. `rsync` the staged export tree back to local
18. import the staged tree into `cluster-results/chapter1_true_results/`
19. regenerate local reports and open the local review notebooks

## Final Expected Local State

At the end of a successful rebuild:

- cluster-side protected working outputs live under the cluster bundle `artifacts/chapter1/`
- staged cluster-to-local review bundles live under the cluster bundle `export-staging/chapter1_true_results/`
- imported approved local-review outputs live under local `cluster-results/chapter1_true_results/`
- local scientific review should prefer `cluster-results/chapter1_true_results/`
- local `artifacts/chapter1/` should be treated as synthetic/development-only unless intentionally regenerated for smoke testing

## If Something Fails

Start with these checks:

1. confirm preprocessing finished and wrote `artifacts/chapter1/`
2. confirm each dependency job finished before launching the next one
3. inspect `logs/*.out` and `logs/*.err` on the cluster
4. confirm `run_chapter1_stage_local_review_exports.py` completed without missing-file errors
5. confirm the local `rsync` copied `export_manifest.json` files into `export-staging/chapter1_true_results/`
6. rerun `python run_chapter1_import_staged_exports.py --overwrite` if the staged files are correct but the local mirror is stale
