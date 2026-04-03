#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with:
# sbatch run_asic_icd10_disease_group_validation.sh
# Prerequisite: upstream ASIC static harmonized artifact exists in hpc-icu-data-platform.
# This writes the ASIC ICD-10 disease-group validation outputs and memo.
#SBATCH --job-name=chapter1_icd10_disease_group_validation
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
script_path="${project_root}/scripts/ch1_asic_icd10_disease_group_inspection.py"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"

mkdir -p "${project_root}/logs"

if [ ! -f "${script_path}" ]; then
    echo "Missing validation script: ${script_path}" >&2
    exit 1
fi

module purge
# module load Python/3.11.5

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

cd "${project_root}"

echo "[$(date)] Starting ASIC ICD-10 disease-group validation job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "ASIC_INPUT_ROOT: ${ASIC_INPUT_ROOT:-auto-resolve}"

python "${script_path}"

echo "[$(date)] ASIC ICD-10 disease-group validation job finished"
