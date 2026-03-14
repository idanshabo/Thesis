#!/bin/bash
#SBATCH --partition=glacier
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=mean_anova_size
#SBATCH --output=pipeline_files/slurm/logs/mean_anova_size_%j.out
#SBATCH --error=pipeline_files/slurm/logs/mean_anova_size_%j.err

eval "$(conda shell.bash hook)"
conda activate kaveret

cd /sci/labs/orzuk/orzuk/github/idan_thesis
mkdir -p pipeline_files/slurm/results pipeline_files/slurm/logs

python pipeline_files/simulate_test_validation.py \
  --test mean_anova \
  --study size \
  --n 200 \
  --p 20 \
  --reps 100 \
  --n_permutations 200 \
  --tree balanced \
  --seed 42 \
  --output pipeline_files/slurm/results/synth_mean_anova_size.json
