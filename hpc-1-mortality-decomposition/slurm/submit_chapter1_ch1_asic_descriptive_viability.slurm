#!/usr/bin/env bash
#SBATCH --job-name=chapter1-ch1-asic-descriptive-viability
#SBATCH --output=logs/chapter1_ch1_asic_descriptive_viability_%j.out
#SBATCH --error=logs/chapter1_ch1_asic_descriptive_viability_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
HARD_CASE_DIR="${HARD_CASE_DIR:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/logistic_regression}"
COMPARISON_DIR="${COMPARISON_DIR:-${HARD_CASE_DIR}/asic_hard_case_comparison}"
HORIZON_FOUNDATION_DIR="${HORIZON_FOUNDATION_DIR:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/horizon_dependence/foundation}"
HORIZON_OVERLAP_DIR="${HORIZON_OVERLAP_DIR:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/horizon_dependence/overlap}"
HORIZON_FINAL_DIR="${HORIZON_FINAL_DIR:-${PROJECT_DIR}/artifacts/chapter1/evaluation/asic/horizon_dependence/final}"
NOTEBOOK_PATH="${NOTEBOOK_PATH:-${PROJECT_DIR}/notebooks/ch1_asic_descriptive_viability_review.ipynb}"
EVIDENCE_PACK_PATH="${EVIDENCE_PACK_PATH:-${PROJECT_DIR}/reports/ch1_asic_descriptive_viability_evidence_pack.md}"
MEMO_PATH="${MEMO_PATH:-${PROJECT_DIR}/reports/ch1_asic_descriptive_viability_memo_draft.md}"

required_paths=(
  "$HARD_CASE_DIR/run_manifest.json"
  "$HARD_CASE_DIR/horizon_hard_case_summary.csv"
  "$COMPARISON_DIR/comparison_table.csv"
  "$COMPARISON_DIR/summary.md"
  "$HORIZON_FOUNDATION_DIR/horizon_summary.csv"
  "$HORIZON_OVERLAP_DIR/pairwise_overlap.csv"
  "$HORIZON_FINAL_DIR/horizon_interpretation_memo.md"
  "$PROJECT_DIR/scripts/run_chapter1_ch1_asic_descriptive_viability.sh"
)

for required_path in "${required_paths[@]}"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing prerequisite artifact or launcher: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/reports"
cd "$PROJECT_DIR"

cmd=(
  "$PROJECT_DIR/scripts/run_chapter1_ch1_asic_descriptive_viability.sh"
  --repo-root "$PROJECT_DIR"
  --notebook-path "$NOTEBOOK_PATH"
  --evidence-pack-path "$EVIDENCE_PACK_PATH"
  --memo-path "$MEMO_PATH"
)

echo "[$(date)] Starting Chapter 1 ASIC descriptive viability-review job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "HARD_CASE_DIR: $HARD_CASE_DIR"
echo "COMPARISON_DIR: $COMPARISON_DIR"
echo "HORIZON_FOUNDATION_DIR: $HORIZON_FOUNDATION_DIR"
echo "HORIZON_OVERLAP_DIR: $HORIZON_OVERLAP_DIR"
echo "HORIZON_FINAL_DIR: $HORIZON_FINAL_DIR"
echo "NOTEBOOK_PATH: $NOTEBOOK_PATH"
echo "EVIDENCE_PACK_PATH: $EVIDENCE_PACK_PATH"
echo "MEMO_PATH: $MEMO_PATH"

"${cmd[@]}"

echo "[$(date)] Chapter 1 ASIC descriptive viability-review job finished"
