#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with: sbatch run_hard_case_agreement.sh
# Run the ASIC logistic baseline and XGBoost recalibration jobs first so saved
# prediction artifacts exist for both models.
# This writes the cross-model ASIC hard-case agreement sensitivity artifacts.
#SBATCH --job-name=chapter1_hard_case_agreement
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
logistic_input_root="${LOGISTIC_INPUT_ROOT:-${project_root}/artifacts/chapter1/baselines/asic/primary_medians}"
xgb_recalibration_root="${XGB_RECALIBRATION_ROOT:-${project_root}/artifacts/chapter1/recalibration/asic/primary_medians/xgboost}"
xgb_recalibration_method="${XGB_RECALIBRATION_METHOD:-platt}"
default_output_dir="${project_root}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/agreement/logistic_regression_vs_xgboost_${xgb_recalibration_method}"
output_dir="${OUTPUT_DIR:-${default_output_dir}}"
horizons="${HORIZONS:-}"
output_format="${OUTPUT_FORMAT:-csv}"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"

mkdir -p "${project_root}/logs"

if [ ! -d "${logistic_input_root}" ]; then
    echo "LOGISTIC_INPUT_ROOT does not exist: ${logistic_input_root}" >&2
    echo "Run the logistic baseline job first, or override LOGISTIC_INPUT_ROOT." >&2
    exit 1
fi

if [ ! -d "${xgb_recalibration_root}" ]; then
    echo "XGB_RECALIBRATION_ROOT does not exist: ${xgb_recalibration_root}" >&2
    echo "Run the XGBoost recalibration job first, or override XGB_RECALIBRATION_ROOT." >&2
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
    run_chapter1_hard_case_agreement.py
    --logistic-input-root "${logistic_input_root}"
    --xgb-recalibration-root "${xgb_recalibration_root}"
    --xgb-recalibration-method "${xgb_recalibration_method}"
    --output-dir "${output_dir}"
    --output-format "${output_format}"
)

if [ -n "${horizons}" ]; then
    # shellcheck disable=SC2206
    horizon_array=(${horizons})
    cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC hard-case agreement job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "LOGISTIC_INPUT_ROOT: ${logistic_input_root}"
echo "XGB_RECALIBRATION_ROOT: ${xgb_recalibration_root}"
echo "XGB_RECALIBRATION_METHOD: ${xgb_recalibration_method}"
echo "OUTPUT_DIR: ${output_dir}"
echo "OUTPUT_FORMAT: ${output_format}"
if [ -n "${horizons}" ]; then
    echo "HORIZONS override: ${horizons}"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC hard-case agreement job finished"
