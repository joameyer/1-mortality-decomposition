#!/usr/bin/env bash
#SBATCH --job-name=chapter1-sofa-feasibility
#SBATCH --output=logs/chapter1_sofa_feasibility_%j.out
#SBATCH --error=logs/chapter1_sofa_feasibility_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL_READY_PATH="${MODEL_READY_PATH:-${PROJECT_DIR}/artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv}"
HARD_CASE_PATH="${HARD_CASE_PATH:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv}"
FEATURE_DEFINITION_PATH="${FEATURE_DEFINITION_PATH:-${PROJECT_DIR}/artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv}"
LOCF_SUMMARY_PATH="${LOCF_SUMMARY_PATH:-${PROJECT_DIR}/artifacts/chapter1/carry_forward/chapter1_primary_locf_feature_summary.csv}"
LAUNCHER_PATH="${PROJECT_DIR}/run_sofa_feasibility.sh"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -f "$MODEL_READY_PATH" ]; then
  echo "MODEL_READY_PATH does not exist: $MODEL_READY_PATH" >&2
  echo "Run preprocessing first, or override MODEL_READY_PATH." >&2
  exit 1
fi

if [ ! -f "$HARD_CASE_PATH" ]; then
  echo "HARD_CASE_PATH does not exist: $HARD_CASE_PATH" >&2
  echo "Run the logistic baseline and hard-case definition jobs first, or override HARD_CASE_PATH." >&2
  exit 1
fi

if [ ! -f "$FEATURE_DEFINITION_PATH" ]; then
  echo "FEATURE_DEFINITION_PATH does not exist: $FEATURE_DEFINITION_PATH" >&2
  echo "Run preprocessing first, or override FEATURE_DEFINITION_PATH." >&2
  exit 1
fi

if [ ! -f "$LOCF_SUMMARY_PATH" ]; then
  echo "LOCF_SUMMARY_PATH does not exist: $LOCF_SUMMARY_PATH" >&2
  echo "Run preprocessing first, or override LOCF_SUMMARY_PATH." >&2
  exit 1
fi

if [ ! -f "$LAUNCHER_PATH" ]; then
  echo "Missing launcher script: $LAUNCHER_PATH" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "[$(date)] Starting ASIC Issue 3.2 SOFA feasibility job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "MODEL_READY_PATH: $MODEL_READY_PATH"
echo "HARD_CASE_PATH: $HARD_CASE_PATH"
echo "FEATURE_DEFINITION_PATH: $FEATURE_DEFINITION_PATH"
echo "LOCF_SUMMARY_PATH: $LOCF_SUMMARY_PATH"

"$LAUNCHER_PATH"

echo "[$(date)] ASIC Issue 3.2 SOFA feasibility job finished"
