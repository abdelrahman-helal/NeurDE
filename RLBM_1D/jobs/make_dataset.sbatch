#!/bin/bash
#SBATCH -J rlbm1d-data
#SBATCH -A <account>
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
source jobs/common.sh
srun python make_dataset.py
