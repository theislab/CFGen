#!/bin/bash

#SBATCH -o ./logs/train_scpoli.out

#SBATCH -e ./logs/train_scpoli.err

#SBATCH -J train_scpoli

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

cd /home/icb/alessandro.palma/environment/cfgen/notebooks/experiments/multi_label /batch_correction /scripts

conda activate celldreamer

python train_scpoli.py