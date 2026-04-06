#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with:
# sbatch run_asic_horizon_dependence_final.sh
# Prerequisites: the horizon foundation and hard-case stability jobs have
# finished successfully.
# This writes the Package 3 mortality-vs-risk comparison figure and memo.
#SBATCH --job-name=chapter1_asic_horizon_final
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
hard_case_dir="${HARD_CASE_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression}"
foundation_dir="${FOUNDATION_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/horizon_dependence/foundation}"
overlap_dir="${OVERLAP_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/horizon_dependence/overlap}"
output_dir="${OUTPUT_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/horizon_dependence/final}"
horizons="${HORIZONS:-}"
bin_count="${BIN_COUNT:-6}"
output_format="${OUTPUT_FORMAT:-csv}"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"

mkdir -p "${project_root}/logs"

if [ ! -f "${hard_case_dir}/run_manifest.json" ]; then
    echo "Missing hard-case manifest: ${hard_case_dir}/run_manifest.json" >&2
    echo "Run the hard-case definition job first, or override HARD_CASE_DIR." >&2
    exit 1
fi

if [ ! -f "${foundation_dir}/horizon_summary.csv" ]; then
    echo "Missing foundation summary: ${foundation_dir}/horizon_summary.csv" >&2
    echo "Run the horizon foundation job first, or override FOUNDATION_DIR." >&2
    exit 1
fi

if [ ! -f "${overlap_dir}/pairwise_overlap.csv" ]; then
    echo "Missing overlap summary: ${overlap_dir}/pairwise_overlap.csv" >&2
    echo "Run the horizon hard-case stability job first, or override OVERLAP_DIR." >&2
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
    run_chapter1_asic_horizon_dependence_final.py
    --hard-case-dir "${hard_case_dir}"
    --foundation-dir "${foundation_dir}"
    --overlap-dir "${overlap_dir}"
    --output-dir "${output_dir}"
    --bin-count "${bin_count}"
    --output-format "${output_format}"
)

if [ -n "${horizons}" ]; then
    # shellcheck disable=SC2206
    horizon_array=(${horizons})
    cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 ASIC final horizon-comparison job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "HARD_CASE_DIR: ${hard_case_dir}"
echo "FOUNDATION_DIR: ${foundation_dir}"
echo "OVERLAP_DIR: ${overlap_dir}"
echo "OUTPUT_DIR: ${output_dir}"
echo "BIN_COUNT: ${bin_count}"
echo "OUTPUT_FORMAT: ${output_format}"
if [ -n "${horizons}" ]; then
    echo "HORIZONS override: ${horizons}"
fi

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC final horizon-comparison job finished"
