#!/bin/bash
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=8                     # Ask for 8 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=30G                             # Ask for 20 GB of RAM
#SBATCH --time=20:00:00                        # The job will run for 9 hours
#SBATCH -o storage/slurm-%j.out  # Write the log on tmp1
#SBATCH --exclude=rtx3,rtx5

# 1. Load your environment
module load anaconda
conda activate conda_env

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python test14.py 