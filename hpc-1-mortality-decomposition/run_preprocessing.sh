#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with: sbatch run_preprocessing.sh
#SBATCH --job-name=chapter1_preprocessing
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
run_config="${RUN_CONFIG:-${project_root}/config/ch1_run_config.json}"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
INPUT_FORMAT="${INPUT_FORMAT:-}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-}"

mkdir -p "${project_root}/logs"

if [ ! -f "${run_config}" ]; then
    echo "RUN_CONFIG does not exist: ${run_config}" >&2
    exit 1
fi

if [ -z "${INPUT_DIR}" ]; then
    artifacts_candidates=(
        "${project_root}/../hpc-icu-data-platform/artifacts"
        "${project_root}/../../hpc-icu-data-platform/artifacts"
    )
    for artifacts_root in "${artifacts_candidates[@]}"; do
        if [ -d "${artifacts_root}/asic_harmonized" ]; then
            INPUT_DIR="${artifacts_root}/asic_harmonized"
            break
        fi
        if [ -d "${artifacts_root}/asic_harmonized_full" ]; then
            INPUT_DIR="${artifacts_root}/asic_harmonized_full"
            break
        fi
    done
fi

module purge
# module load Python/3.11.5

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

cd "${project_root}"

cmd=(
    python
    run_chapter1_preprocessing.py
    --run-config "${run_config}"
)

if [ -n "${INPUT_DIR}" ]; then
    cmd+=(--input-dir "${INPUT_DIR}")
fi
if [ -n "${OUTPUT_DIR}" ]; then
    cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [ -n "${INPUT_FORMAT}" ]; then
    cmd+=(--input-format "${INPUT_FORMAT}")
fi
if [ -n "${OUTPUT_FORMAT}" ]; then
    cmd+=(--output-format "${OUTPUT_FORMAT}")
fi

echo "[$(date)] Starting Chapter 1 preprocessing job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "RUN_CONFIG: ${run_config}"
if [ -n "${INPUT_DIR}" ]; then
    echo "INPUT_DIR override: ${INPUT_DIR}"
fi
if [ -n "${OUTPUT_DIR}" ]; then
    echo "OUTPUT_DIR override: ${OUTPUT_DIR}"
fi
if [ -n "${INPUT_FORMAT}" ]; then
    echo "INPUT_FORMAT override: ${INPUT_FORMAT}"
fi
if [ -n "${OUTPUT_FORMAT}" ]; then
    echo "OUTPUT_FORMAT override: ${OUTPUT_FORMAT}"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 preprocessing job finished"
