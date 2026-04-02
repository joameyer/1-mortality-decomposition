#!/usr/bin/env bash
#SBATCH --job-name=chapter1-temporal-preview
#SBATCH --output=logs/chapter1_temporal_preview_%j.out
#SBATCH --error=logs/chapter1_temporal_preview_%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_CONFIG="${RUN_CONFIG:-${PROJECT_DIR}/config/ch1_run_config.json}"
INPUT_DIR="${INPUT_DIR:-}"
INPUT_FORMAT="${INPUT_FORMAT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/artifacts/chapter1/temporal_preview/asic/aggregation_16h}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-csv}"
FROZEN_CHAPTER1_DIR="${FROZEN_CHAPTER1_DIR:-${PROJECT_DIR}/artifacts/chapter1}"
EIGHT_HOUR_EVALUATION_ROOT="${EIGHT_HOUR_EVALUATION_ROOT:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/baselines/primary_medians}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-${PROJECT_DIR}/notebooks/ch1_asic_temporal_aggregation_preview_16h.ipynb}"
BLOCK_HOURS="${BLOCK_HOURS:-16}"
HORIZONS="${HORIZONS:-}"

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

if [ ! -d "$EIGHT_HOUR_EVALUATION_ROOT" ]; then
  echo "EIGHT_HOUR_EVALUATION_ROOT does not exist: $EIGHT_HOUR_EVALUATION_ROOT" >&2
  echo "Run the frozen 8h evaluation first, or override EIGHT_HOUR_EVALUATION_ROOT." >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/run_chapter1_temporal_preview.sh" ]; then
  echo "Missing launcher script: $PROJECT_DIR/scripts/run_chapter1_temporal_preview.sh" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# Optional environment setup for your cluster:
# module load Python/3.11.5
# source .venv/bin/activate

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_temporal_preview.sh"
  --run-config "$RUN_CONFIG"
  --output-root "$OUTPUT_ROOT"
  --output-format "$OUTPUT_FORMAT"
  --frozen-chapter1-dir "$FROZEN_CHAPTER1_DIR"
  --eight-hour-evaluation-root "$EIGHT_HOUR_EVALUATION_ROOT"
  --notebook-path "$NOTEBOOK_PATH"
  --block-hours "$BLOCK_HOURS"
)

if [ -n "$INPUT_DIR" ]; then
  cmd+=(--input-dir "$INPUT_DIR")
fi
if [ -n "$INPUT_FORMAT" ]; then
  cmd+=(--input-format "$INPUT_FORMAT")
fi
if [ -n "$HORIZONS" ]; then
  # shellcheck disable=SC2206
  horizon_array=($HORIZONS)
  cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC 16h temporal aggregation preview"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "RUN_CONFIG: $RUN_CONFIG"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "OUTPUT_FORMAT: $OUTPUT_FORMAT"
echo "FROZEN_CHAPTER1_DIR: $FROZEN_CHAPTER1_DIR"
echo "EIGHT_HOUR_EVALUATION_ROOT: $EIGHT_HOUR_EVALUATION_ROOT"
echo "NOTEBOOK_PATH: $NOTEBOOK_PATH"
echo "BLOCK_HOURS: $BLOCK_HOURS"
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

echo "[$(date)] Chapter 1 ASIC 16h temporal aggregation preview finished"
