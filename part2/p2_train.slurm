#!/bin/bash
#SBATCH --job-name=p2_train
#SBATCH --account=torch_pr_932_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/my3281/1011-a4/part2/logs/%x_%j.out

LR=$1
NAME=$2
cd /scratch/$USER/1011-a4/part2
source /scratch/$USER/hw4p2.sh
echo "LR=$LR NAME=$NAME"
python3 train_t5.py --finetune --learning_rate $LR \
  --scheduler_type cosine --num_warmup_epochs 1 \
  --max_n_epochs 20 --patience_epochs 4 \
  --batch_size 16 --test_batch_size 32 \
  --experiment_name $NAME
