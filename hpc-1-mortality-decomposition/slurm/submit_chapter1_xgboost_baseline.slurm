#!/usr/bin/env bash
#SBATCH --job-name=chapter1-xgboost-baseline
#SBATCH --output=logs/chapter1_xgboost_baseline_%j.out
#SBATCH --error=logs/chapter1_xgboost_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
INPUT_DATASET="${INPUT_DATASET:-${PROJECT_DIR}/artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv}"
FEATURE_SET_DEFINITION="${FEATURE_SET_DEFINITION:-${PROJECT_DIR}/artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/artifacts/chapter1/baselines/asic/primary_medians/xgboost}"
HORIZONS="${HORIZONS:-}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -f "$INPUT_DATASET" ]; then
  echo "INPUT_DATASET does not exist: $INPUT_DATASET" >&2
  echo "Run preprocessing first, or override INPUT_DATASET." >&2
  exit 1
fi

if [ ! -f "$FEATURE_SET_DEFINITION" ]; then
  echo "FEATURE_SET_DEFINITION does not exist: $FEATURE_SET_DEFINITION" >&2
  echo "Run preprocessing first, or override FEATURE_SET_DEFINITION." >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/run_chapter1_xgboost_baseline.sh" ]; then
  echo "Missing launcher script: $PROJECT_DIR/scripts/run_chapter1_xgboost_baseline.sh" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# Optional environment setup for your cluster:
# module load Python/3.11.5
# source .venv/bin/activate

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_xgboost_baseline.sh"
  --input-dataset "$INPUT_DATASET"
  --feature-set-definition "$FEATURE_SET_DEFINITION"
  --output-dir "$OUTPUT_DIR"
)

if [ -n "$HORIZONS" ]; then
  # shellcheck disable=SC2206
  horizon_array=($HORIZONS)
  cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC XGBoost baseline job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "INPUT_DATASET: $INPUT_DATASET"
echo "FEATURE_SET_DEFINITION: $FEATURE_SET_DEFINITION"
echo "OUTPUT_DIR: $OUTPUT_DIR"
if [ -n "$HORIZONS" ]; then
  echo "HORIZONS override: $HORIZONS"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC XGBoost baseline job finished"
