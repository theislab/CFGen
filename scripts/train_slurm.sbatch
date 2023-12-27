#!/bin/bash

#SBATCH -o ./logs/train_slurm.out

#SBATCH -e ./logs/train_slurm.err

#SBATCH -J train_slurm

#SBATCH -p gpu_all

#SBATCH --qos=phd

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer

python ../celldreamer/train.py denoising_module=resnet