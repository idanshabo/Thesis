#!/bin/bash
# ============================================================================
# Simulation validation — SLURM launcher.
#
# Usage:
#   bash run_simulation.sh                              # submit ALL (9 jobs)
#   bash run_simulation.sh all                          # same as above
#   bash run_simulation.sh mean_anova size              # single job
#   bash run_simulation.sh mean_anova power             # single job
#   bash run_simulation.sh cov_lrt size real /path.tree # single job with real tree
#
# Arguments:
#   $1 = test:  mean_anova | mean_lrt | cov_lrt | all   (default: all)
#   $2 = study: size | power | robustness | grid | all  (default: all)
#   $3 = tree:  balanced | random | real                 (default: balanced)
#   $4 = tree path (required if $3=real)
# ============================================================================

REPO_DIR="/sci/labs/orzuk/orzuk/github/idan_thesis"
SCRIPT="${REPO_DIR}/pipeline_files/simulate_test_validation.py"
RESULTS_DIR="${REPO_DIR}/pipeline_files/slurm/results"
LOGS_DIR="${REPO_DIR}/pipeline_files/slurm/logs"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

TEST="${1:-all}"
STUDY="${2:-all}"
TREE="${3:-balanced}"
TREE_PATH="${4:-}"

# Build tree args
TREE_ARGS="--tree ${TREE}"
if [ "${TREE}" = "real" ] && [ -n "${TREE_PATH}" ]; then
    TREE_ARGS="${TREE_ARGS} --tree_path ${TREE_PATH}"
fi

# Expand "all" into lists
if [ "${TEST}" = "all" ]; then
    TESTS="mean_anova mean_lrt cov_lrt"
else
    TESTS="${TEST}"
fi

if [ "${STUDY}" = "all" ]; then
    STUDIES="size power robustness"
else
    STUDIES="${STUDY}"
fi

# Submit one sbatch job per (test, study) combination
for t in ${TESTS}; do
    for s in ${STUDIES}; do
        JOB_NAME="sim_${t}_${s}"
        OUTPUT="${RESULTS_DIR}/${t}_${s}_${TREE}.json"

        sbatch \
          --partition=glacier \
          --time=02:00:00 \
          --mem=4G \
          --cpus-per-task=4 \
          --job-name="${JOB_NAME}" \
          --output="${LOGS_DIR}/${JOB_NAME}_%j.out" \
          --error="${LOGS_DIR}/${JOB_NAME}_%j.err" \
          <<SLURM_EOF
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate kaveret
cd ${REPO_DIR}

python ${SCRIPT} \
  --test ${t} \
  --study ${s} \
  --n 200 \
  --p 20 \
  --reps 100 \
  --n_permutations 200 \
  --n_bootstrap 200 \
  --seed 42 \
  --plot \
  ${TREE_ARGS} \
  --output ${OUTPUT}
SLURM_EOF

        echo "Submitted: ${JOB_NAME}"
    done
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in:   ${RESULTS_DIR}/"
echo "Logs in:      ${LOGS_DIR}/"
