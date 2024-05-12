#!/bin/bash

#SBATCH -o ./logs/tabula_muris.out

#SBATCH -e ./logs/tabula_muris.err

#SBATCH -J tabula_muris

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=160G

#SBATCH -t 2-00:00

#SBATCH --nice=10000
c
conda activate celldreamer

export HYDRA_FULL_ERROR=1

python ../compute_metrics.py --dataset_name 'tabula_muris' --category_name 'tissue' --batched \
--batch_size 2000 --n_batches 10 --nn 10

