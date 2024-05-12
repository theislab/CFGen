#!/bin/bash

#SBATCH -o ./logs/c_elegans.out

#SBATCH -e ./logs/c_elegans.err

#SBATCH -J c_elegans

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer

export HYDRA_FULL_ERROR=1

python ../compute_metrics_per_cell_type.py --dataset_name 'c_elegans' --category_name 'cell_type' --nn 10
