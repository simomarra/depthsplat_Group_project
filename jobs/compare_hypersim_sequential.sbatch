#!/bin/bash
#SBATCH --job-name=ds-hypersim-9
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/3274075/DepthSplat_view_selection/logs/hypersim-sequential-%j.log

set -euo pipefail
cd /home/3274075/DepthSplat_view_selection/code_primary_attention

# Run each scene in a fresh process inside this single GPU allocation.
# The original array script is executed by Bash; its #SBATCH lines are ignored.
export SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID"
for index in {1..9}; do
    export SLURM_ARRAY_TASK_ID="$index"
    printf '\nSTART SCENE %s OF 9\n' "$index"
    bash jobs/compare_hypersim_remaining.sbatch
    printf 'FINISHED SCENE %s OF 9\n' "$index"
done
printf 'ALL NINE HYPERSIM SCENES COMPLETE\n'
