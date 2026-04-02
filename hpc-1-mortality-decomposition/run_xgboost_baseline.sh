#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with: sbatch run_xgboost_baseline.sh
# Run preprocessing first so artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv exists.
# Run after installing xgboost in the active environment.
#SBATCH --job-name=chapter1_xgboost_baseline
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
input_dataset="${INPUT_DATASET:-${project_root}/artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv}"
feature_set_definition="${FEATURE_SET_DEFINITION:-${project_root}/artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv}"
output_dir="${OUTPUT_DIR:-${project_root}/artifacts/chapter1/baselines/asic/primary_medians/xgboost}"
horizons="${HORIZONS:-}"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"

mkdir -p "${project_root}/logs"

if [ ! -f "${input_dataset}" ]; then
    echo "INPUT_DATASET does not exist: ${input_dataset}" >&2
    echo "Run preprocessing first, or override INPUT_DATASET." >&2
    exit 1
fi

if [ ! -f "${feature_set_definition}" ]; then
    echo "FEATURE_SET_DEFINITION does not exist: ${feature_set_definition}" >&2
    echo "Run preprocessing first, or override FEATURE_SET_DEFINITION." >&2
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
    run_chapter1_xgboost_baseline.py
    --input-dataset "${input_dataset}"
    --feature-set-definition "${feature_set_definition}"
    --output-dir "${output_dir}"
)

if [ -n "${horizons}" ]; then
    # shellcheck disable=SC2206
    horizon_array=(${horizons})
    cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC XGBoost baseline job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "INPUT_DATASET: ${input_dataset}"
echo "FEATURE_SET_DEFINITION: ${feature_set_definition}"
echo "OUTPUT_DIR: ${output_dir}"
if [ -n "${horizons}" ]; then
    echo "HORIZONS override: ${horizons}"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC XGBoost baseline job finished"
