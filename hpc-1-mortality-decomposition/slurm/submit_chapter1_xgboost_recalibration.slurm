#!/usr/bin/env bash
#SBATCH --job-name=chapter1-xgboost-recalibration
#SBATCH --output=logs/chapter1_xgboost_recalibration_%j.out
#SBATCH --error=logs/chapter1_xgboost_recalibration_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_DIR}/artifacts/chapter1/baselines/asic/primary_medians}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/artifacts/chapter1/recalibration/asic/primary_medians/xgboost}"
HORIZONS="${HORIZONS:-}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -d "$INPUT_ROOT" ]; then
  echo "INPUT_ROOT does not exist: $INPUT_ROOT" >&2
  echo "Run the baseline model jobs first, or override INPUT_ROOT." >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/run_chapter1_xgboost_recalibration.sh" ]; then
  echo "Missing launcher script: $PROJECT_DIR/scripts/run_chapter1_xgboost_recalibration.sh" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# Optional environment setup for your cluster:
# module load Python/3.11.5
# source .venv/bin/activate

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_xgboost_recalibration.sh"
  --input-root "$INPUT_ROOT"
  --output-dir "$OUTPUT_DIR"
)

if [ -n "$HORIZONS" ]; then
  # shellcheck disable=SC2206
  horizon_array=($HORIZONS)
  cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC XGBoost recalibration job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "INPUT_ROOT: $INPUT_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
if [ -n "$HORIZONS" ]; then
  echo "HORIZONS override: $HORIZONS"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC XGBoost recalibration job finished"
