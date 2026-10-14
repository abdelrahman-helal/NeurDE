#!/bin/bash
#SBATCH -J rlbm1d-stage1
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
source jobs/common.sh
srun python train_stage_1.py
