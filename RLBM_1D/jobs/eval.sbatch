#!/bin/bash
#SBATCH -J rlbm1d-eval
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
source jobs/common.sh
srun python eval.py --classical --checkpoint checkpoints/stage1/best.pt --out results/eval_stage1
srun python eval.py --classical --checkpoint checkpoints/stage2/best.pt --out results/eval_stage2
