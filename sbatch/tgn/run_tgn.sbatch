#!/bin/sh
#SBATCH --job-name=ml_optuna
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --error=results/tgn-optuna-%A-%a.err
#SBATCH --output=results/tgn-optuna-%A-%a.log
#SBATCH --time=6:00:00
#SBATCH --array=21-40

MODEL=tgn

echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "Model = $MODEL"

OPTUNA_PARAMS="tgn"
NUM_TRIALS=200
N_STARTUP_TRIALS=10

module load Python
conda activate recsys-btl

python3 sbatch/run.py \
    --dataset_idx "$SLURM_ARRAY_TASK_ID" \
    --model "$MODEL" \
    --n_jobs 12 \
    --optuna_params "$OPTUNA_PARAMS" \
    --num_trials "$NUM_TRIALS" \
    --n_startup_trials "$N_STARTUP_TRIALS" \