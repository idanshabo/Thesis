#!/bin/bash
# ============================================================================
# Simulation validation — SLURM launcher.
#
# Usage:
#   bash run_simulation.sh                                    # submit ALL synthetic (9 jobs)
#   bash run_simulation.sh all                                # same as above
#   bash run_simulation.sh mean_anova size                    # single job, synthetic tree
#   bash run_simulation.sh mean_anova size real PF00076       # single job, real tree from family
#   bash run_simulation.sh mean_anova size real /path/to.tree # single job, real tree from path
#   bash run_simulation.sh all all real PF00076               # all 9 jobs with real tree
#
# Arguments:
#   $1 = test:   mean_anova | mean_lrt | cov_lrt | all   (default: all)
#   $2 = study:  size | power | robustness | grid | all  (default: all)
#   $3 = tree:   balanced | random | real                 (default: balanced)
#   $4 = family or tree path (when $3=real):
#        - Pfam ID like PF00076 -> auto-resolves paths from config
#        - /full/path/to/file.tree -> uses path directly
# ============================================================================

REPO_DIR="/sci/labs/orzuk/orzuk/github/idan_thesis"
SCRIPT="${REPO_DIR}/pipeline_files/simulate_test_validation.py"
RESULTS_DIR="${REPO_DIR}/pipeline_files/slurm/results"
LOGS_DIR="${REPO_DIR}/pipeline_files/slurm/logs"

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

TEST="${1:-all}"
STUDY="${2:-all}"
TREE="${3:-balanced}"
FAMILY_OR_PATH="${4:-}"

# Build tree args
TREE_ARGS="--tree ${TREE}"
TREE_LABEL="${TREE}"
if [ "${TREE}" = "real" ] && [ -n "${FAMILY_OR_PATH}" ]; then
    if [[ "${FAMILY_OR_PATH}" == /* ]]; then
        # Absolute path given
        TREE_ARGS="${TREE_ARGS} --tree_path ${FAMILY_OR_PATH}"
        TREE_LABEL="real_custom"
    else
        # Pfam family ID given — let Python resolve via config
        TREE_ARGS="${TREE_ARGS} --family ${FAMILY_OR_PATH}"
        TREE_LABEL="real_${FAMILY_OR_PATH}"
    fi
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
        OUTPUT="${RESULTS_DIR}/${t}_${s}_${TREE_LABEL}.json"

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
