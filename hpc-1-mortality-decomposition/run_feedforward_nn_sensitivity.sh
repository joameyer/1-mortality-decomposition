#!/bin/bash

# SLURM directives
# Submit this script from inside hpc-1-mortality-decomposition/ with:
#   sbatch run_feedforward_nn_sensitivity.sh
#
# Default mode is an isolated 24h feedforward-NN baseline check.
# To run the bounded sensitivity wrapper instead, submit with:
#   sbatch --export=ALL,RUN_MODE=sensitivity24 run_feedforward_nn_sensitivity.sh
#
# After the 24h check succeeds, expand horizons with for example:
#   sbatch --export=ALL,RUN_MODE=sensitivity_full,HORIZONS="8 16 24 48 72" run_feedforward_nn_sensitivity.sh
#
# Optional cluster-specific override:
#   sbatch --partition=c23ms run_feedforward_nn_sensitivity.sh
#
#SBATCH --job-name=chapter1_feedforward_nn
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# #SBATCH --account=rwth1641

set -euo pipefail

project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
input_dataset="${INPUT_DATASET:-${project_root}/artifacts/chapter1/model_ready/chapter1_primary_model_ready_dataset.csv}"
feature_set_definition="${FEATURE_SET_DEFINITION:-${project_root}/artifacts/chapter1/feature_sets/chapter1_feature_set_definition.csv}"
baseline_input_root="${BASELINE_INPUT_ROOT:-${project_root}/artifacts/chapter1/baselines/asic/primary_medians}"
nn_output_dir="${NN_OUTPUT_DIR:-${project_root}/artifacts/chapter1/baselines/asic/primary_medians/feedforward_nn}"
evaluation_output_dir="${EVALUATION_OUTPUT_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/baselines/primary_medians}"
hard_case_output_dir="${HARD_CASE_OUTPUT_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/feedforward_nn}"
sensitivity_output_dir="${SENSITIVITY_OUTPUT_DIR:-${project_root}/artifacts/chapter1/evaluation/asic/hard_cases/primary_medians/agreement/feedforward_nn_sensitivity}"
run_mode="${RUN_MODE:-baseline24}"
horizons="${HORIZONS:-24}"

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

if [ ! -d "${baseline_input_root}" ]; then
    echo "BASELINE_INPUT_ROOT does not exist: ${baseline_input_root}" >&2
    echo "Run the frozen baseline jobs first, or override BASELINE_INPUT_ROOT." >&2
    exit 1
fi

module purge
# module load Python/3.11.5

if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

cd "${project_root}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

case "${run_mode}" in
    baseline24|baseline|baseline_full)
        cmd=(
            python
            run_chapter1_feedforward_nn_baseline.py
            --input-dataset "${input_dataset}"
            --feature-set-definition "${feature_set_definition}"
            --output-dir "${nn_output_dir}"
        )
        ;;
    sensitivity24|sensitivity|sensitivity_full)
        cmd=(
            python
            run_chapter1_feedforward_nn_hard_case_sensitivity.py
            --baseline-input-root "${baseline_input_root}"
            --nn-output-dir "${nn_output_dir}"
            --evaluation-output-dir "${evaluation_output_dir}"
            --hard-case-output-dir "${hard_case_output_dir}"
            --sensitivity-output-dir "${sensitivity_output_dir}"
            --input-dataset "${input_dataset}"
            --feature-set-definition "${feature_set_definition}"
        )
        ;;
    *)
        echo "Unsupported RUN_MODE: ${run_mode}" >&2
        echo "Expected one of: baseline24, baseline, baseline_full, sensitivity24, sensitivity, sensitivity_full" >&2
        exit 1
        ;;
esac

if [ -n "${horizons}" ]; then
    # shellcheck disable=SC2206
    horizon_array=(${horizons})
    cmd+=(--horizons "${horizon_array[@]}")
fi

echo "[$(date)] Starting Chapter 1 feedforward-NN batch job"
echo "HOSTNAME: $(hostname)"
echo "PROJECT_ROOT: ${project_root}"
echo "RUN_MODE: ${run_mode}"
echo "HORIZONS: ${horizons}"
echo "INPUT_DATASET: ${input_dataset}"
echo "FEATURE_SET_DEFINITION: ${feature_set_definition}"
echo "BASELINE_INPUT_ROOT: ${baseline_input_root}"
echo "NN_OUTPUT_DIR: ${nn_output_dir}"
echo "EVALUATION_OUTPUT_DIR: ${evaluation_output_dir}"
echo "HARD_CASE_OUTPUT_DIR: ${hard_case_output_dir}"
echo "SENSITIVITY_OUTPUT_DIR: ${sensitivity_output_dir}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS}"
echo "VECLIB_MAXIMUM_THREADS: ${VECLIB_MAXIMUM_THREADS}"
echo "Command: ${cmd[*]}"

"${cmd[@]}"

echo "[$(date)] Chapter 1 feedforward-NN batch job finished"
