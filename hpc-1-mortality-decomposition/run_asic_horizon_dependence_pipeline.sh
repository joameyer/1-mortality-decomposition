#!/usr/bin/env bash

set -euo pipefail

project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ ! -d "${project_root}" ]; then
    echo "PROJECT_ROOT does not exist: ${project_root}" >&2
    exit 1
fi

for required_script in \
    run_asic_horizon_dependence_foundation.sh \
    run_asic_horizon_hard_case_stability.sh \
    run_asic_horizon_dependence_final.sh
do
    if [ ! -f "${project_root}/${required_script}" ]; then
        echo "Missing required script: ${project_root}/${required_script}" >&2
        exit 1
    fi
done

cd "${project_root}"

echo "[$(date)] Submitting Chapter 1 ASIC horizon-dependence pipeline"
echo "PROJECT_ROOT: ${project_root}"

jid1=$(sbatch --parsable --export=ALL "${project_root}/run_asic_horizon_dependence_foundation.sh")
jid2=$(sbatch --parsable --export=ALL --dependency=afterok:"${jid1}" "${project_root}/run_asic_horizon_hard_case_stability.sh")
jid3=$(sbatch --parsable --export=ALL --dependency=afterok:"${jid2}" "${project_root}/run_asic_horizon_dependence_final.sh")

echo "foundation=${jid1} overlap=${jid2} final=${jid3}"
echo
echo "Follow-up checks:"
echo "  squeue -u am861154"
echo "  ls -lt logs | head"
