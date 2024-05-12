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

python ../compute_metrics.py --dataset_name 'pbmc68k' --category_name 'celltype' --batched \
--batch_size 2000 --n_batches 10 --nn 10

