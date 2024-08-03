#!/bin/bash

#SBATCH -o ./logs/pbmc_rebuttals.out

#SBATCH -e ./logs/pbmc_rebuttals.err

#SBATCH -J pbmc_rebuttals

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000



conda activate celldreamer

export HYDRA_FULL_ERROR=1

python ../compute_metrics_multimodal_pbmc.py --category_name 'cell_type' \
--nn 10
