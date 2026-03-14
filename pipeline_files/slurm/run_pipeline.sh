#!/bin/bash
# ============================================================================
# Run the full pipeline on one or more Pfam families via SLURM.
#
# Usage:
#   bash run_pipeline.sh PF00321                    # single family
#   bash run_pipeline.sh PF00321 PF01340 PF14549    # multiple families
#   bash run_pipeline.sh all                        # all 4 of Idan's families
#
# Arguments:
#   $@ = family IDs (or "all" for PF00321 PF01340 PF14549 PF16168)
# ============================================================================

REPO_DIR="/sci/labs/orzuk/orzuk/github/idan_thesis"
DATA_DIR="/sci/labs/orzuk/orzuk/projects/kaveret/data"
SCRIPT="${REPO_DIR}/pipeline_files/full_pipeline_mean_shift.py"
LOGS_DIR="${REPO_DIR}/pipeline_files/slurm/logs"

mkdir -p "${LOGS_DIR}"

# Expand "all"
if [ "$1" = "all" ] || [ -z "$1" ]; then
    FAMILIES="PF00321 PF01340 PF14549 PF16168"
else
    FAMILIES="$@"
fi

for fam in ${FAMILIES}; do
    MSA="${DATA_DIR}/${fam}.stockholm"
    if [ ! -f "${MSA}" ]; then
        echo "Warning: ${MSA} not found, skipping ${fam}"
        continue
    fi

    sbatch \
      --partition=glacier \
      --time=04:00:00 \
      --mem=16G \
      --cpus-per-task=4 \
      --job-name="pipe_${fam}" \
      --output="${LOGS_DIR}/pipe_${fam}_%j.out" \
      --error="${LOGS_DIR}/pipe_${fam}_%j.err" \
      <<SLURM_EOF
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate kaveret
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH}"

python ${SCRIPT} \
  --input ${MSA} \
  --family ${fam} \
  --operation find_best_split \
  --embedding sequence \
  --generate_plots FALSE
SLURM_EOF

    echo "Submitted: pipe_${fam}"
done

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in:      ${LOGS_DIR}/"
