#!/bin/bash

#SBATCH -o ./logs/pbmc3k.out

#SBATCH -e ./logs/pbmc3k.err

#SBATCH -J pbmc3k

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000



conda activate celldreamer

export HYDRA_FULL_ERROR=1

python ../compute_metrics.py --dataset_name 'pbmc3k' --category_name 'cell_type' --nn 10
