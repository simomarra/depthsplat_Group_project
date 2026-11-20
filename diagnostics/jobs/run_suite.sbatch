#!/bin/bash -l
#SBATCH --job-name=depth_diagnostics
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics_%j.out

set -euo pipefail
# Submit from repository root. Supply the existing activated environment, or set
# DIAG_ENV to the conda prefix; module names follow the recorded Bocconi setup.
: "${SUITE_REPO:?Set SUITE_REPO to the isolated checkout}"
if [[ -n "${DIAG_ENV:-}" ]]; then
    module load miniconda3
    module load cuda/12.4
    source /software/miniconda3/etc/profile.d/conda.sh
    conda activate "$DIAG_ENV"
fi
cd "$SUITE_REPO"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTHONPATH="$SUITE_REPO${PYTHONPATH:+:$PYTHONPATH}"
exec python -m diagnostics.run "$@"
