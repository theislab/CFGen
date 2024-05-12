#!/bin/bash

#SBATCH -o ./logs/pbmc68k.out

#SBATCH -e ./logs/pbmc68k.err

#SBATCH -J pbmc68k

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000



conda activate celldreamer

export HYDRA_FULL_ERROR=1

python ../compute_metrics_per_cell_type.py --dataset_name 'pbmc68k' --category_name 'celltype' --nn 10
