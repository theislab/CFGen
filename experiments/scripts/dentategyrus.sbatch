#!/bin/bash

#SBATCH -o ./logs/dentategyrus.out

#SBATCH -e ./logs/dentategyrus.err

#SBATCH -J dentategyrus

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer

export HYDRA_FULL_ERROR=1

python ../compute_metrics.py --dataset_name 'dentategyrus' --category_name 'clusters' --nn 10
