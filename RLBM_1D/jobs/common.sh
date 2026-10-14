# Sourced by the job scripts: environment and working directory on Perlmutter.
module load conda
eval "$(conda shell.bash hook)"
conda activate NeurDE              # or: conda activate /path/to/envs/neurde
cd "$SLURM_SUBMIT_DIR"             # submit from NeurDE/RLBM_1D
