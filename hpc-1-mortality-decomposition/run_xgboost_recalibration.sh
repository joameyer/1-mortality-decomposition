#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with: sbatch run_xgboost_recalibration.sh
# Run the logistic and XGBoost baseline jobs first so saved prediction artifacts exist.
# This fits recalibration on validation only and applies the fitted mappings to test.
#SBATCH --job-name=chapter1_xgboost_recalibration
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
input_root="${INPUT_ROOT:-${project_root}/artifacts/chapter1/baselines/asic/primary_medians}"
output_dir="${OUTPUT_DIR:-${project_root}/artifacts/chapter1/recalibration/asic/primary_medians/xgboost}"
horizons="${HORIZONS:-}"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"

mkdir -p "${project_root}/logs"

if [ ! -d "${input_root}" ]; then
    echo "INPUT_ROOT does not exist: ${input_root}" >&2
    echo "Run the baseline model jobs first, or override INPUT_ROOT." >&2
    exit 1
fi

module purge
# module load Python/3.11.5

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

cd "${project_root}"

cmd=(
    python
    run_chapter1_xgboost_recalibration.py
    --input-root "${input_root}"
    --output-dir "${output_dir}"
)

if [ -n "${horizons}" ]; then
    # shellcheck disable=SC2206
    horizon_array=(${horizons})
    cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC XGBoost recalibration job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "INPUT_ROOT: ${input_root}"
echo "OUTPUT_DIR: ${output_dir}"
if [ -n "${horizons}" ]; then
    echo "HORIZONS override: ${horizons}"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC XGBoost recalibration job finished"
