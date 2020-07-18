#!/bin/bash
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=10G                             # Ask for 10 GB of RAM
#SBATCH --time=24:00:00                        # The job will run for 9 hours
#SBATCH -o storage/slurm-%j.out  # Write the log on tmp1
#SBATCH --exclude=eos[18-21]

# 1. Load your environment
module load anaconda
conda activate default

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python examples.py