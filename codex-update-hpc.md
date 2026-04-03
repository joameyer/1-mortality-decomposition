Inside this repo, `/Users/joanameyer/repository/1-mortality-decomposition`, there is a nested HPC upload bundle at:

`/Users/joanameyer/repository/1-mortality-decomposition/hpc-1-mortality-decomposition`

Context:
- The main local repo is where I develop and test code against a small synthetic sample dataset.
- The nested `hpc-1-mortality-decomposition` directory is the bundle I upload to the RWTH HPC cluster to run the real analysis on the full dataset.
- When I ask this, I want you to update the HPC bundle so it matches the relevant new code/notebook changes from the main repo.

What I want you to do:
1. Inspect the recent/local changes in the main repo.
2. Propagate all relevant code, notebook, script, config, and documentation changes into `hpc-1-mortality-decomposition`.
3. Preserve any HPC-specific adjustments where needed, and do not copy over local-only clutter such as `.git`, `.venv`, `__pycache__`, `.ipynb_checkpoints`, logs, artifacts, or sample-data-only outputs unless explicitly needed.
4. If something is ambiguous, make the safest reasonable assumption and state it briefly.
5. After updating the HPC bundle, give me:
   - a short summary of what you changed in the HPC bundle
   - the exact upload command I should run from my terminal
   - the exact command(s) I should run on the cluster after upload to execute the most recent changes
   - a short note on which directory on the cluster I should `cd` into before running them

For the cluster-run guidance:
- Infer the correct job entrypoint from the files you updated in the HPC bundle.
- Prefer the real bundle-specific command, usually something like `sbatch run_preprocessing.sh` or `sbatch run_xgboost_baseline.sh`, not a generic placeholder.
- If multiple jobs should be run, list them in the correct order.
- If there are important follow-up checks, include them briefly, for example `squeue -u am861154` or checking the `logs/` directory.

The upload command should be this unless you found a concrete reason it must differ:

`rsync -avh --progress \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'logs/' \
  --exclude 'artifacts/' \
  /Users/joanameyer/repository/1-mortality-decomposition/hpc-1-mortality-decomposition/ \
  am861154@login23-1.hpc.itc.rwth-aachen.de:/home/am861154/projects/hpc-1-mortality-decomposition/`

Please actually make the HPC bundle edits in this workspace, then show me the final rsync command.
