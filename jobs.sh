#!/bin/bash
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=8                     # Ask for 8 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=20G                             # Ask for 20 GB of RAM
#SBATCH --time=24:00:00                        # The job will run for 9 hours
#SBATCH -o storage/slurm-%j.out  # Write the log on tmp1
#SBATCH --exclude=rtx3,rtx5,leto20


cd ~
module load python/3.6
source venv-deeprl-py36/bin/activate
cd projects/successor-feature-representations

python retrain_examples_v2.py