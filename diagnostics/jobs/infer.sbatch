#!/bin/bash -l
#SBATCH --job-name=secondary_export
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=secondary_export_%j.out
set -euo pipefail
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
exec python -m diagnostics.infer "$@"
