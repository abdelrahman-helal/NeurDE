#!/bin/bash
# Submit the whole pipeline; each job starts only if the previous one succeeded.
# Run from NeurDE/RLBM_1D:  bash jobs/submit_all.sh
set -e
mkdir -p logs
data=$(sbatch --parsable jobs/make_dataset.sbatch)
s1=$(sbatch --parsable --dependency=afterok:$data jobs/stage1.sbatch)
s2=$(sbatch --parsable --dependency=afterok:$s1 jobs/stage2.sbatch)
ev=$(sbatch --parsable --dependency=afterok:$s2 jobs/eval.sbatch)
echo "submitted: data $data -> stage1 $s1 -> stage2 $s2 -> eval $ev"
