#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with:
# sbatch run_ch1_asic_descriptive_viability.sh
# This job only regenerates the review notebook and memo/report outputs from
# already-saved ASIC hard-case and horizon-dependence artifacts.
# Run the hard-case comparison and horizon-dependence jobs first if those
# prerequisite artifacts are stale or missing.
#SBATCH --job-name=chapter1_ch1_asic_descriptive_viability
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
hard_case_dir="${HARD_CASE_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression}"
comparison_dir="${COMPARISON_DIR:-${hard_case_dir}/asic_hard_case_comparison}"
horizon_foundation_dir="${HORIZON_FOUNDATION_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/horizon_dependence/foundation}"
horizon_overlap_dir="${HORIZON_OVERLAP_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/horizon_dependence/overlap}"
horizon_final_dir="${HORIZON_FINAL_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/horizon_dependence/final}"
notebook_path="${NOTEBOOK_PATH:-${project_root}/notebooks/ch1_asic_descriptive_viability_review.ipynb}"
evidence_pack_path="${EVIDENCE_PACK_PATH:-${project_root}/reports/ch1_asic_descriptive_viability_evidence_pack.md}"
memo_path="${MEMO_PATH:-${project_root}/reports/ch1_asic_descriptive_viability_memo_draft.md}"

VENV_PATH="${VENV_PATH:-/home/am861154/projects/hpc-1-mortality-decomposition/.venv}"

mkdir -p "${project_root}/logs" "${project_root}/reports"

required_paths=(
    "${hard_case_dir}/run_manifest.json"
    "${hard_case_dir}/horizon_hard_case_summary.csv"
    "${comparison_dir}/comparison_table.csv"
    "${comparison_dir}/summary.md"
    "${horizon_foundation_dir}/horizon_summary.csv"
    "${horizon_overlap_dir}/pairwise_overlap.csv"
    "${horizon_final_dir}/horizon_interpretation_memo.md"
)

for required_path in "${required_paths[@]}"; do
    if [ ! -f "${required_path}" ]; then
        echo "Missing prerequisite artifact: ${required_path}" >&2
        echo "Refresh the upstream hard-case comparison and horizon-dependence jobs first." >&2
        exit 1
    fi
done

module purge
# module load Python/3.11.5

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

cd "${project_root}"

cmd=(
    python
    run_chapter1_ch1_asic_descriptive_viability.py
    --repo-root "${project_root}"
    --notebook-path "${notebook_path}"
    --evidence-pack-path "${evidence_pack_path}"
    --memo-path "${memo_path}"
)

echo "[$(date)] Starting Chapter 1 ASIC descriptive viability-review job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "HARD_CASE_DIR: ${hard_case_dir}"
echo "COMPARISON_DIR: ${comparison_dir}"
echo "HORIZON_FOUNDATION_DIR: ${horizon_foundation_dir}"
echo "HORIZON_OVERLAP_DIR: ${horizon_overlap_dir}"
echo "HORIZON_FINAL_DIR: ${horizon_final_dir}"
echo "NOTEBOOK_PATH: ${notebook_path}"
echo "EVIDENCE_PACK_PATH: ${evidence_pack_path}"
echo "MEMO_PATH: ${memo_path}"

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC descriptive viability-review job finished"
