#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with: sbatch run_temporal_preview.sh
# Run the frozen 8h preprocessing, baseline model jobs, and 8h baseline evaluation first.
# This builds a separate 16h preview package and compares it back to the saved 8h outputs.
#SBATCH --job-name=chapter1_temporal_preview
#SBATCH --time=36:00:00
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
output_root="${OUTPUT_ROOT:-${project_root}/artifacts/chapter1/temporal_preview/asic/aggregation_16h}"
output_format="${OUTPUT_FORMAT:-csv}"
frozen_chapter1_dir="${FROZEN_CHAPTER1_DIR:-${project_root}/artifacts/chapter1}"
eight_hour_evaluation_root="${EIGHT_HOUR_EVALUATION_ROOT:-${project_root}/artifacts/chapter1/evaluation/asic/baselines/primary_medians}"
notebook_path="${NOTEBOOK_PATH:-${project_root}/notebooks/ch1_asic_temporal_aggregation_preview_16h.ipynb}"
block_hours="${BLOCK_HOURS:-16}"
horizons="${HORIZONS:-}"

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

if [ ! -d "${frozen_chapter1_dir}" ]; then
    echo "FROZEN_CHAPTER1_DIR does not exist: ${frozen_chapter1_dir}" >&2
    exit 1
fi

if [ ! -f "${frozen_chapter1_dir}/splits/chapter1_stay_split_assignments.csv" ]; then
    echo "Missing frozen split assignments: ${frozen_chapter1_dir}/splits/chapter1_stay_split_assignments.csv" >&2
    echo "Run the frozen 8h preprocessing first, or override FROZEN_CHAPTER1_DIR." >&2
    exit 1
fi

if [ ! -d "${eight_hour_evaluation_root}" ]; then
    echo "EIGHT_HOUR_EVALUATION_ROOT does not exist: ${eight_hour_evaluation_root}" >&2
    echo "Run the frozen 8h evaluation first, or override EIGHT_HOUR_EVALUATION_ROOT." >&2
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
    run_chapter1_temporal_preview.py
    --run-config "${run_config}"
    --output-root "${output_root}"
    --output-format "${output_format}"
    --frozen-chapter1-dir "${frozen_chapter1_dir}"
    --eight-hour-evaluation-root "${eight_hour_evaluation_root}"
    --notebook-path "${notebook_path}"
    --block-hours "${block_hours}"
)

if [ -n "${input_dir}" ]; then
    cmd+=(--input-dir "${input_dir}")
fi
if [ -n "${input_format}" ]; then
    cmd+=(--input-format "${input_format}")
fi
if [ -n "${horizons}" ]; then
    # shellcheck disable=SC2206
    horizon_array=(${horizons})
    cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC 16h temporal aggregation preview"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "RUN_CONFIG: ${run_config}"
echo "OUTPUT_ROOT: ${output_root}"
echo "OUTPUT_FORMAT: ${output_format}"
echo "FROZEN_CHAPTER1_DIR: ${frozen_chapter1_dir}"
echo "EIGHT_HOUR_EVALUATION_ROOT: ${eight_hour_evaluation_root}"
echo "NOTEBOOK_PATH: ${notebook_path}"
echo "BLOCK_HOURS: ${block_hours}"
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

echo "[$(date)] Chapter 1 ASIC 16h temporal aggregation preview finished"
