#!/bin/bash

#SBATCH -J diff_hlca
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=6
#SBATCH --nice=10000


source $HOME/.bashrc
conda activate celldreamer
cd ..
cd ..
cd celldreamer/trainer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/icb/till.richter/anaconda3/envs/celldreamer/lib
python -u hlca.py