#!/usr/bin/env bash
#SBATCH --job-name=chapter1-preprocessing
#SBATCH --output=logs/chapter1_preprocessing_%j.out
#SBATCH --error=logs/chapter1_preprocessing_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_CONFIG="${RUN_CONFIG:-${PROJECT_DIR}/config/ch1_run_config.json}"
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
INPUT_FORMAT="${INPUT_FORMAT:-}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-}"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -f "$RUN_CONFIG" ]; then
  echo "RUN_CONFIG does not exist: $RUN_CONFIG" >&2
  exit 1
fi

if [ ! -f "$PROJECT_DIR/scripts/run_chapter1_preprocessing.sh" ]; then
  echo "Missing launcher script: $PROJECT_DIR/scripts/run_chapter1_preprocessing.sh" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# Optional environment setup for your cluster:
# module load Python/3.11.5
# source .venv/bin/activate

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_preprocessing.sh"
  --run-config "$RUN_CONFIG"
)

if [ -n "$INPUT_DIR" ]; then
  cmd+=(--input-dir "$INPUT_DIR")
fi
if [ -n "$OUTPUT_DIR" ]; then
  cmd+=(--output-dir "$OUTPUT_DIR")
fi
if [ -n "$INPUT_FORMAT" ]; then
  cmd+=(--input-format "$INPUT_FORMAT")
fi
if [ -n "$OUTPUT_FORMAT" ]; then
  cmd+=(--output-format "$OUTPUT_FORMAT")
fi

echo "[$(date)] Starting Chapter 1 preprocessing job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "RUN_CONFIG: $RUN_CONFIG"
if [ -n "$INPUT_DIR" ]; then
  echo "INPUT_DIR override: $INPUT_DIR"
fi
if [ -n "$OUTPUT_DIR" ]; then
  echo "OUTPUT_DIR override: $OUTPUT_DIR"
fi
if [ -n "$INPUT_FORMAT" ]; then
  echo "INPUT_FORMAT override: $INPUT_FORMAT"
fi
if [ -n "$OUTPUT_FORMAT" ]; then
  echo "OUTPUT_FORMAT override: $OUTPUT_FORMAT"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 preprocessing job finished"
