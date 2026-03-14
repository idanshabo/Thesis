#!/bin/bash
#============================================================================
# KAVERET Simulation Validation — SLURM launcher
#
# Submits independent jobs for each (test × study) combination.
# Each job writes JSON results + plots to cluster/results/.
#
# Usage:
#   cd /path/to/Thesis
#   bash cluster/run_simulations.sh          # submit all jobs
#   bash cluster/run_simulations.sh quick     # fast sanity check (~5 min)
#============================================================================

set -euo pipefail

# ---- Configuration (edit these for your HUJI cluster) ----
PARTITION="${PARTITION:-gpu-a100-killable}"   # or: short, long, gpu-a100
CONDA_ENV="${CONDA_ENV:-thesis}"             # your conda environment name
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
RESULTS_DIR="${REPO_DIR}/cluster/results"
SCRIPT="${REPO_DIR}/pipeline_files/simulate_test_validation.py"

mkdir -p "${RESULTS_DIR}" "${REPO_DIR}/cluster/logs"

# ---- Simulation parameters ----
MODE="${1:-full}"

if [ "$MODE" = "quick" ]; then
    # Sanity check: ~5 min total
    REPS=50
    BOOT=100
    PERM=200
    TIME="00:30:00"
    MEM="4G"
    echo "=== QUICK MODE (sanity check) ==="
else
    # Full validation for paper: ~2-6 hours per job
    REPS=500
    BOOT=1000
    PERM=999
    TIME="12:00:00"
    MEM="8G"
    echo "=== FULL MODE (paper results) ==="
fi

echo "  Repo:       ${REPO_DIR}"
echo "  Results:    ${RESULTS_DIR}"
echo "  Conda env:  ${CONDA_ENV}"
echo "  Partition:  ${PARTITION}"
echo ""

# ---- Helper: submit one job ----
submit_job() {
    local JOB_NAME="$1"
    local CMD="$2"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=kav_${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
#SBATCH --output=${REPO_DIR}/cluster/logs/${JOB_NAME}_%j.out
#SBATCH --error=${REPO_DIR}/cluster/logs/${JOB_NAME}_%j.err

echo "Job \${SLURM_JOB_ID} started at \$(date)"
echo "Node: \$(hostname)"
echo "Command: ${CMD}"

# Load environment
module load anaconda 2>/dev/null || true
source activate ${CONDA_ENV} 2>/dev/null || conda activate ${CONDA_ENV}

cd ${REPO_DIR}
${CMD}

echo "Job \${SLURM_JOB_ID} finished at \$(date)"
EOF

    echo "  Submitted: ${JOB_NAME}"
}

# ====================================================================
# Job 1: Covariance LRT — Type I error
# ====================================================================
submit_job "cov_size" \
    "python ${SCRIPT} --test cov_lrt --study size \
        --n 200 --p 20 --reps ${REPS} --n_bootstrap ${BOOT} \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/cov_lrt_size.json"

# ====================================================================
# Job 2: Covariance LRT — Power curve
# ====================================================================
submit_job "cov_power" \
    "python ${SCRIPT} --test cov_lrt --study power \
        --shift_type covariance --n 200 --p 20 --reps ${REPS} \
        --n_bootstrap ${BOOT} --delta_scales 0.1 0.2 0.5 1.0 2.0 5.0 \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/cov_lrt_power.json"

# ====================================================================
# Job 3: Covariance LRT — OU robustness
# ====================================================================
submit_job "cov_robust" \
    "python ${SCRIPT} --test cov_lrt --study robustness \
        --n 200 --p 20 --reps ${REPS} --n_bootstrap ${BOOT} \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/cov_lrt_robustness.json"

# ====================================================================
# Job 4: Covariance LRT — (n, p) grid
# ====================================================================
submit_job "cov_grid" \
    "python ${SCRIPT} --test cov_lrt --study grid \
        --reps ${REPS} --n_bootstrap ${BOOT} \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/cov_lrt_grid.json"

# ====================================================================
# Job 5: Mean ANOVA — Type I error
# ====================================================================
submit_job "mean_size" \
    "python ${SCRIPT} --test mean_anova --study size \
        --n 200 --p 20 --reps ${REPS} --n_permutations ${PERM} \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/mean_anova_size.json"

# ====================================================================
# Job 6: Mean ANOVA — Power curve
# ====================================================================
submit_job "mean_power" \
    "python ${SCRIPT} --test mean_anova --study power \
        --shift_type mean --n 200 --p 20 --reps ${REPS} \
        --n_permutations ${PERM} --delta_scales 0.1 0.2 0.5 1.0 2.0 5.0 \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/mean_anova_power.json"

# ====================================================================
# Job 7: Mean ANOVA — OU robustness
# ====================================================================
submit_job "mean_robust" \
    "python ${SCRIPT} --test mean_anova --study robustness \
        --n 200 --p 20 --reps ${REPS} --n_permutations ${PERM} \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/mean_anova_robustness.json"

# ====================================================================
# Job 8: Cross-sensitivity — does cov LRT detect mean shifts?
# ====================================================================
submit_job "cross_cov_mean" \
    "python ${SCRIPT} --test cov_lrt --study power \
        --shift_type mean --n 200 --p 20 --reps ${REPS} \
        --n_bootstrap ${BOOT} --delta_scales 0.5 1.0 2.0 5.0 \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/cross_cov_detects_mean.json"

# ====================================================================
# Job 9: Cross-sensitivity — does mean ANOVA detect cov shifts?
# ====================================================================
submit_job "cross_mean_cov" \
    "python ${SCRIPT} --test mean_anova --study power \
        --shift_type covariance --n 200 --p 20 --reps ${REPS} \
        --n_permutations ${PERM} --delta_scales 0.5 1.0 2.0 5.0 \
        --tree balanced --seed 42 --plot \
        --output ${RESULTS_DIR}/cross_mean_detects_cov.json"

echo ""
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
echo "Results will appear in: ${RESULTS_DIR}/"
echo "Logs in: ${REPO_DIR}/cluster/logs/"
