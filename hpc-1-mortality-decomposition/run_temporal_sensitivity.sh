#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with: sbatch run_temporal_sensitivity.sh
# Prerequisites:
# - frozen 8h preprocessing has written artifacts/chapter1/
# - frozen 8h baseline evaluation has written artifacts/chapter1/evaluation/asic/baselines/primary_medians/
# - frozen 8h logistic hard-case definition has written
#   artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression/
# This job rebuilds 16h and 24h coarsened aggregations and writes the formal Chapter 1 temporal
# sensitivity package under artifacts/chapter1/temporal_sensitivity/asic/.
#SBATCH --job-name=chapter1_temporal_sensitivity
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
run_config="${RUN_CONFIG:-${project_root}/config/ch1_run_config.json}"
input_dir="${INPUT_DIR:-}"
input_format="${INPUT_FORMAT:-}"
output_root="${OUTPUT_ROOT:-${project_root}/artifacts/chapter1/temporal_sensitivity/asic}"
output_format="${OUTPUT_FORMAT:-csv}"
frozen_chapter1_dir="${FROZEN_CHAPTER1_DIR:-${project_root}/artifacts/chapter1}"
reference_evaluation_root="${REFERENCE_EVALUATION_ROOT:-${project_root}/artifacts/chapter1/evaluation/asic/baselines/primary_medians}"
reference_hard_case_root="${REFERENCE_HARD_CASE_ROOT:-${project_root}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression}"
sensitivity_block_hours="${SENSITIVITY_BLOCK_HOURS:-16 24}"
horizons="${HORIZONS:-}"
models="${MODELS:-logistic_regression xgboost}"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"

mkdir -p "${project_root}/logs"

if [ ! -f "${run_config}" ]; then
    echo "RUN_CONFIG does not exist: ${run_config}" >&2
    exit 1
fi

if [ -z "${input_dir}" ]; then
    artifacts_candidates=(
        "${project_root}/../hpc-icu-data-platform/artifacts"
        "${project_root}/../../hpc-icu-data-platform/artifacts"
    )
    for artifacts_root in "${artifacts_candidates[@]}"; do
        if [ -d "${artifacts_root}/asic_harmonized" ]; then
            input_dir="${artifacts_root}/asic_harmonized"
            break
        fi
        if [ -d "${artifacts_root}/asic_harmonized_full" ]; then
            input_dir="${artifacts_root}/asic_harmonized_full"
            break
        fi
    done
fi

if [ ! -f "${frozen_chapter1_dir}/splits/chapter1_stay_split_assignments.csv" ]; then
    echo "Missing frozen split assignments: ${frozen_chapter1_dir}/splits/chapter1_stay_split_assignments.csv" >&2
    echo "Run the frozen 8h preprocessing first, or override FROZEN_CHAPTER1_DIR." >&2
    exit 1
fi

if [ ! -d "${reference_evaluation_root}" ]; then
    echo "REFERENCE_EVALUATION_ROOT does not exist: ${reference_evaluation_root}" >&2
    echo "Run the frozen 8h baseline evaluation first, or override REFERENCE_EVALUATION_ROOT." >&2
    exit 1
fi

if [ ! -f "${reference_hard_case_root}/run_manifest.json" ]; then
    echo "Missing frozen logistic hard-case manifest: ${reference_hard_case_root}/run_manifest.json" >&2
    echo "Run the frozen 8h hard-case definition first, or override REFERENCE_HARD_CASE_ROOT." >&2
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
    run_chapter1_temporal_sensitivity.py
    --run-config "${run_config}"
    --output-root "${output_root}"
    --output-format "${output_format}"
    --frozen-chapter1-dir "${frozen_chapter1_dir}"
    --reference-evaluation-root "${reference_evaluation_root}"
    --reference-hard-case-root "${reference_hard_case_root}"
)

if [ -n "${input_dir}" ]; then
    cmd+=(--input-dir "${input_dir}")
fi
if [ -n "${input_format}" ]; then
    cmd+=(--input-format "${input_format}")
fi
if [ -n "${sensitivity_block_hours}" ]; then
    # shellcheck disable=SC2206
    aggregation_array=(${sensitivity_block_hours})
    cmd+=(--sensitivity-block-hours "${aggregation_array[@]}")
fi
if [ -n "${horizons}" ]; then
    # shellcheck disable=SC2206
    horizon_array=(${horizons})
    cmd+=(--horizons "${horizon_array[@]}")
fi
if [ -n "${models}" ]; then
    # shellcheck disable=SC2206
    model_array=(${models})
    cmd+=(--models "${model_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC temporal aggregation sensitivity analysis"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "RUN_CONFIG: ${run_config}"
echo "OUTPUT_ROOT: ${output_root}"
echo "OUTPUT_FORMAT: ${output_format}"
echo "FROZEN_CHAPTER1_DIR: ${frozen_chapter1_dir}"
echo "REFERENCE_EVALUATION_ROOT: ${reference_evaluation_root}"
echo "REFERENCE_HARD_CASE_ROOT: ${reference_hard_case_root}"
echo "SENSITIVITY_BLOCK_HOURS: ${sensitivity_block_hours}"
echo "MODELS: ${models}"
if [ -n "${input_dir}" ]; then
    echo "INPUT_DIR override: ${input_dir}"
fi
if [ -n "${input_format}" ]; then
    echo "INPUT_FORMAT override: ${input_format}"
fi
if [ -n "${horizons}" ]; then
    echo "HORIZONS override: ${horizons}"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC temporal aggregation sensitivity analysis finished"
