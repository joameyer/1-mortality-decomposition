#!/usr/bin/env bash
#SBATCH --job-name=chapter1-asic-icd10-disease-group-validation
#SBATCH --output=logs/chapter1_asic_icd10_disease_group_validation_%j.out
#SBATCH --error=logs/chapter1_asic_icd10_disease_group_validation_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
LAUNCHER_PATH="${PROJECT_DIR}/run_asic_icd10_disease_group_validation.sh"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if [ ! -f "$LAUNCHER_PATH" ]; then
  echo "Missing launcher script: $LAUNCHER_PATH" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "[$(date)] Starting ASIC ICD-10 disease-group validation job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "ASIC_INPUT_ROOT: ${ASIC_INPUT_ROOT:-auto-resolve}"

"$LAUNCHER_PATH"

echo "[$(date)] ASIC ICD-10 disease-group validation job finished"
