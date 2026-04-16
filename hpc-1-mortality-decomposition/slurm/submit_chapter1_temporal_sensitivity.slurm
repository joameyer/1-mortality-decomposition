#!/usr/bin/env bash
#SBATCH --job-name=chapter1-temporal-sensitivity
#SBATCH --output=logs/chapter1_temporal_sensitivity_%j.out
#SBATCH --error=logs/chapter1_temporal_sensitivity_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_CONFIG="${RUN_CONFIG:-${PROJECT_DIR}/config/ch1_run_config.json}"
INPUT_DIR="${INPUT_DIR:-}"
INPUT_FORMAT="${INPUT_FORMAT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/chapter1/temporal_sensitivity/asic}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-csv}"
FROZEN_CHAPTER1_DIR="${FROZEN_CHAPTER1_DIR:-${PROJECT_DIR}/artifacts/chapter1}"
REFERENCE_EVALUATION_ROOT="${REFERENCE_EVALUATION_ROOT:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/baselines/primary_medians}"
REFERENCE_HARD_CASE_ROOT="${REFERENCE_HARD_CASE_ROOT:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression}"
SENSITIVITY_BLOCK_HOURS="${SENSITIVITY_BLOCK_HOURS:-16 24}"
HORIZONS="${HORIZONS:-}"
MODELS="${MODELS:-logistic_regression xgboost}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -f "$RUN_CONFIG" ]; then
  echo "RUN_CONFIG does not exist: $RUN_CONFIG" >&2
  exit 1
fi

if [ ! -f "$FROZEN_CHAPTER1_DIR/splits/chapter1_stay_split_assignments.csv" ]; then
  echo "Missing frozen split assignments: $FROZEN_CHAPTER1_DIR/splits/chapter1_stay_split_assignments.csv" >&2
  echo "Run the frozen 8h preprocessing first, or override FROZEN_CHAPTER1_DIR." >&2
  exit 1
fi

if [ ! -d "$REFERENCE_EVALUATION_ROOT" ]; then
  echo "REFERENCE_EVALUATION_ROOT does not exist: $REFERENCE_EVALUATION_ROOT" >&2
  echo "Run the frozen 8h evaluation first, or override REFERENCE_EVALUATION_ROOT." >&2
  exit 1
fi

if [ ! -f "$REFERENCE_HARD_CASE_ROOT/run_manifest.json" ]; then
  echo "Missing frozen logistic hard-case manifest: $REFERENCE_HARD_CASE_ROOT/run_manifest.json" >&2
  echo "Run the frozen 8h hard-case definition first, or override REFERENCE_HARD_CASE_ROOT." >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/run_chapter1_temporal_sensitivity.sh" ]; then
  echo "Missing launcher script: $PROJECT_DIR/scripts/run_chapter1_temporal_sensitivity.sh" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_temporal_sensitivity.sh"
  --run-config "$RUN_CONFIG"
  --output-root "$OUTPUT_ROOT"
  --output-format "$OUTPUT_FORMAT"
  --frozen-chapter1-dir "$FROZEN_CHAPTER1_DIR"
  --reference-evaluation-root "$REFERENCE_EVALUATION_ROOT"
  --reference-hard-case-root "$REFERENCE_HARD_CASE_ROOT"
)

if [ -n "$INPUT_DIR" ]; then
  cmd+=(--input-dir "$INPUT_DIR")
fi
if [ -n "$INPUT_FORMAT" ]; then
  cmd+=(--input-format "$INPUT_FORMAT")
fi
if [ -n "$SENSITIVITY_BLOCK_HOURS" ]; then
  # shellcheck disable=SC2206
  aggregation_array=($SENSITIVITY_BLOCK_HOURS)
  cmd+=(--sensitivity-block-hours "${aggregation_array[@]}")
fi
if [ -n "$HORIZONS" ]; then
  # shellcheck disable=SC2206
  horizon_array=($HORIZONS)
  cmd+=(--horizons "${horizon_array[@]}")
fi
if [ -n "$MODELS" ]; then
  # shellcheck disable=SC2206
  model_array=($MODELS)
  cmd+=(--models "${model_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC temporal aggregation sensitivity analysis"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "RUN_CONFIG: $RUN_CONFIG"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "OUTPUT_FORMAT: $OUTPUT_FORMAT"
echo "FROZEN_CHAPTER1_DIR: $FROZEN_CHAPTER1_DIR"
echo "REFERENCE_EVALUATION_ROOT: $REFERENCE_EVALUATION_ROOT"
echo "REFERENCE_HARD_CASE_ROOT: $REFERENCE_HARD_CASE_ROOT"
echo "SENSITIVITY_BLOCK_HOURS: $SENSITIVITY_BLOCK_HOURS"
echo "MODELS: $MODELS"
if [ -n "$INPUT_DIR" ]; then
  echo "INPUT_DIR override: $INPUT_DIR"
fi
if [ -n "$INPUT_FORMAT" ]; then
  echo "INPUT_FORMAT override: $INPUT_FORMAT"
fi
if [ -n "$HORIZONS" ]; then
  echo "HORIZONS override: $HORIZONS"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC temporal aggregation sensitivity analysis finished"
