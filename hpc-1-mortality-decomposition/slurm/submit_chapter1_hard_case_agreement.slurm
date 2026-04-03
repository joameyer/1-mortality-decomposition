#!/usr/bin/env bash
#SBATCH --job-name=chapter1-hard-case-agreement
#SBATCH --output=logs/chapter1_hard_case_agreement_%j.out
#SBATCH --error=logs/chapter1_hard_case_agreement_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
LOGISTIC_INPUT_ROOT="${LOGISTIC_INPUT_ROOT:-${PROJECT_DIR}/artifacts/chapter1/baselines/asic/primary_medians}"
XGB_RECALIBRATION_ROOT="${XGB_RECALIBRATION_ROOT:-${PROJECT_DIR}/artifacts/chapter1/recalibration/asic/primary_medians/xgboost}"
XGB_RECALIBRATION_METHOD="${XGB_RECALIBRATION_METHOD:-platt}"
DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_${XGB_RECALIBRATION_METHOD}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
HORIZONS="${HORIZONS:-}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-csv}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -d "$LOGISTIC_INPUT_ROOT" ]; then
  echo "LOGISTIC_INPUT_ROOT does not exist: $LOGISTIC_INPUT_ROOT" >&2
  echo "Run the logistic baseline job first, or override LOGISTIC_INPUT_ROOT." >&2
  exit 1
fi

if [ ! -d "$XGB_RECALIBRATION_ROOT" ]; then
  echo "XGB_RECALIBRATION_ROOT does not exist: $XGB_RECALIBRATION_ROOT" >&2
  echo "Run the XGBoost recalibration job first, or override XGB_RECALIBRATION_ROOT." >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/run_chapter1_hard_case_agreement.sh" ]; then
  echo "Missing launcher script: $PROJECT_DIR/scripts/run_chapter1_hard_case_agreement.sh" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# Optional environment setup for your cluster:
# module load Python/3.11.5
# source .venv/bin/activate

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_hard_case_agreement.sh"
  --logistic-input-root "$LOGISTIC_INPUT_ROOT"
  --xgb-recalibration-root "$XGB_RECALIBRATION_ROOT"
  --xgb-recalibration-method "$XGB_RECALIBRATION_METHOD"
  --output-dir "$OUTPUT_DIR"
  --output-format "$OUTPUT_FORMAT"
)

if [ -n "$HORIZONS" ]; then
  # shellcheck disable=SC2206
  horizon_array=($HORIZONS)
  cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC hard-case agreement job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "LOGISTIC_INPUT_ROOT: $LOGISTIC_INPUT_ROOT"
echo "XGB_RECALIBRATION_ROOT: $XGB_RECALIBRATION_ROOT"
echo "XGB_RECALIBRATION_METHOD: $XGB_RECALIBRATION_METHOD"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "OUTPUT_FORMAT: $OUTPUT_FORMAT"
if [ -n "$HORIZONS" ]; then
  echo "HORIZONS override: $HORIZONS"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC hard-case agreement job finished"
