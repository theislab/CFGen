#!/bin/bash

#SBATCH -o ./logs/neurips.out

#SBATCH -e ./logs/neurips.err

#SBATCH -J neurips

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000



conda activate celldreamer

export HYDRA_FULL_ERROR=1

python ../compute_metrics.py --dataset_name 'neurips' --category_name 'cell_type' --batched \
--batch_size 1000 --n_batches 10 --nn 10
