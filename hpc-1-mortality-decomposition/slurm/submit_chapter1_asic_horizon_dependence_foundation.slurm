#!/usr/bin/env bash
#SBATCH --job-name=chapter1-asic-horizon-foundation
#SBATCH --output=logs/chapter1_asic_horizon_dependence_foundation_%j.out
#SBATCH --error=logs/chapter1_asic_horizon_dependence_foundation_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
HARD_CASE_DIR="${HARD_CASE_DIR:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/horizon_dependence/foundation}"
HORIZONS="${HORIZONS:-}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-csv}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -f "$HARD_CASE_DIR/run_manifest.json" ]; then
  echo "Missing hard-case manifest: $HARD_CASE_DIR/run_manifest.json" >&2
  echo "Run the hard-case definition job first, or override HARD_CASE_DIR." >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/run_chapter1_asic_horizon_dependence_foundation.sh" ]; then
  echo "Missing launcher script: $PROJECT_DIR/scripts/run_chapter1_asic_horizon_dependence_foundation.sh" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_asic_horizon_dependence_foundation.sh"
  --hard-case-dir "$HARD_CASE_DIR"
  --output-dir "$OUTPUT_DIR"
  --output-format "$OUTPUT_FORMAT"
)

if [ -n "$HORIZONS" ]; then
  # shellcheck disable=SC2206
  horizon_array=($HORIZONS)
  cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC horizon-dependence foundation job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "HARD_CASE_DIR: $HARD_CASE_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "OUTPUT_FORMAT: $OUTPUT_FORMAT"
if [ -n "$HORIZONS" ]; then
  echo "HORIZONS override: $HORIZONS"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC horizon-dependence foundation job finished"
