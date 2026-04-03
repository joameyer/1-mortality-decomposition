#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with:
# sbatch run_sofa_feasibility.sh
# Prerequisites: preprocessing + logistic baseline + hard-case definition.
# This writes the ASIC Issue 3.2 SOFA feasibility memo and component table.
#SBATCH --job-name=chapter1_sofa_feasibility
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
model_ready_path="${MODEL_READY_PATH:-${project_root}/artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv}"
hard_case_path="${HARD_CASE_PATH:-${project_root}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/stay_level_hard_case_flags.csv}"
feature_definition_path="${FEATURE_DEFINITION_PATH:-${project_root}/artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv}"
locf_summary_path="${LOCF_SUMMARY_PATH:-${project_root}/artifacts/chapter1/carry_forward/chapter1_primary_locf_feature_summary.csv}"
script_path="${project_root}/scripts/ch1_sofa_feasibility_audit.py"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"

mkdir -p "${project_root}/logs"

if [ ! -f "${model_ready_path}" ]; then
    echo "MODEL_READY_PATH does not exist: ${model_ready_path}" >&2
    echo "Run preprocessing first, or override MODEL_READY_PATH." >&2
    exit 1
fi

if [ ! -f "${hard_case_path}" ]; then
    echo "HARD_CASE_PATH does not exist: ${hard_case_path}" >&2
    echo "Run the logistic baseline and hard-case definition jobs first, or override HARD_CASE_PATH." >&2
    exit 1
fi

if [ ! -f "${feature_definition_path}" ]; then
    echo "FEATURE_DEFINITION_PATH does not exist: ${feature_definition_path}" >&2
    echo "Run preprocessing first, or override FEATURE_DEFINITION_PATH." >&2
    exit 1
fi

if [ ! -f "${locf_summary_path}" ]; then
    echo "LOCF_SUMMARY_PATH does not exist: ${locf_summary_path}" >&2
    echo "Run preprocessing first, or override LOCF_SUMMARY_PATH." >&2
    exit 1
fi

if [ ! -f "${script_path}" ]; then
    echo "Missing feasibility script: ${script_path}" >&2
    exit 1
fi

module purge
# module load Python/3.11.5

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

cd "${project_root}"

echo "[$(date)] Starting ASIC Issue 3.2 SOFA feasibility job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "MODEL_READY_PATH: ${model_ready_path}"
echo "HARD_CASE_PATH: ${hard_case_path}"
echo "FEATURE_DEFINITION_PATH: ${feature_definition_path}"
echo "LOCF_SUMMARY_PATH: ${locf_summary_path}"

python "${script_path}"

echo "[$(date)] ASIC Issue 3.2 SOFA feasibility job finished"
