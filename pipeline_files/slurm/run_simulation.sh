#!/bin/bash
#SBATCH --partition=glacier
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sim_validation
#SBATCH --output=pipeline_files/slurm/logs/sim_%j.out
#SBATCH --error=pipeline_files/slurm/logs/sim_%j.err

# ============================================================================
# Generic simulation validation script.
#
# Usage examples:
#   sbatch run_simulation.sh                          # defaults: mean_anova, Type I error
#   sbatch run_simulation.sh mean_anova power         # mean ANOVA power curve
#   sbatch run_simulation.sh cov_lrt size             # covariance LRT Type I error
#   sbatch run_simulation.sh mean_lrt robustness      # mean LRT OU robustness
#
# Arguments:
#   $1 = test name:  mean_anova | mean_lrt | cov_lrt       (default: mean_anova)
#   $2 = study type: size | power | robustness | grid       (default: size)
#   $3 = tree type:  balanced | random | real               (default: balanced)
#   $4 = tree path:  /path/to/file.tree (required if $3=real)
# ============================================================================

eval "$(conda shell.bash hook)"
conda activate kaveret

cd /sci/labs/orzuk/orzuk/github/idan_thesis
mkdir -p pipeline_files/slurm/results pipeline_files/slurm/logs

TEST="${1:-mean_anova}"
STUDY="${2:-size}"
TREE="${3:-balanced}"
TREE_PATH="${4:-}"

TREE_ARGS="--tree ${TREE}"
if [ "${TREE}" = "real" ] && [ -n "${TREE_PATH}" ]; then
    TREE_ARGS="${TREE_ARGS} --tree_path ${TREE_PATH}"
fi

OUTPUT="pipeline_files/slurm/results/${TEST}_${STUDY}_${TREE}.json"

python pipeline_files/simulate_test_validation.py \
  --test "${TEST}" \
  --study "${STUDY}" \
  --n 200 \
  --p 20 \
  --reps 100 \
  --n_permutations 200 \
  --n_bootstrap 200 \
  --seed 42 \
  --plot \
  ${TREE_ARGS} \
  --output "${OUTPUT}"
