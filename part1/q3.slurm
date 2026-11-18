#!/bin/bash
#SBATCH --job-name=q3_aug
#SBATCH --account=torch_pr_932_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

cd /scratch/$USER/1011-a4/part1
source /scratch/$USER/hw4p1.sh
python3 main.py --train_augmented --eval_transformed
python3 main.py --eval --model_dir out_augmented
python3 main.py --eval_transformed --model_dir out_augmented
