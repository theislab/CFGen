#!/bin/bash

#SBATCH -o ./logs/hlca_core.out

#SBATCH -e ./logs/hlca_core.err

#SBATCH -J hlca_core

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=300G

#SBATCH -t 2-00:00

#SBATCH --nice=10000


conda activate celldreamer

export HYDRA_FULL_ERROR=1

python ../compute_metrics_per_cell_type.py --dataset_name 'hlca_core' --category_name 'cell_type' --nn 10