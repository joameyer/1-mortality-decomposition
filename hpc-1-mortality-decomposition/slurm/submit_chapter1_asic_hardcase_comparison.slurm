#!/usr/bin/env bash
#SBATCH --job-name=chapter1-asic-hardcase-comparison
#SBATCH --output=logs/chapter1_asic_hardcase_comparison_%j.out
#SBATCH --error=logs/chapter1_asic_hardcase_comparison_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MODEL_READY_PATH="${MODEL_READY_PATH:-${PROJECT_DIR}/artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv}"
HARD_CASE_PATH="${HARD_CASE_PATH:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/issue_3_2_asic_hardcase_comparison}"
ASIC_INPUT_ROOT="${ASIC_INPUT_ROOT:-}"

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

if [ ! -f "$PROJECT_DIR/scripts/run_chapter1_asic_hardcase_comparison.sh" ]; then
  echo "Missing launcher script: $PROJECT_DIR/scripts/run_chapter1_asic_hardcase_comparison.sh" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_asic_hardcase_comparison.sh"
  --hard-case-path "$HARD_CASE_PATH"
  --model-ready-path "$MODEL_READY_PATH"
  --output-dir "$OUTPUT_DIR"
)

if [ -n "$ASIC_INPUT_ROOT" ]; then
  cmd+=(--asic-input-root "$ASIC_INPUT_ROOT")
fi

echo "[$(date)] Starting Chapter 1 ASIC Issue 3.2 hard-case comparison job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "MODEL_READY_PATH: $MODEL_READY_PATH"
echo "HARD_CASE_PATH: $HARD_CASE_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
if [ -n "$ASIC_INPUT_ROOT" ]; then
  echo "ASIC_INPUT_ROOT override: $ASIC_INPUT_ROOT"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC Issue 3.2 hard-case comparison job finished"
