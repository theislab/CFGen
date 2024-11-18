#!/bin/bash

#SBATCH -o ./logs/neurips.out

#SBATCH -e ./logs/neurips.err

#SBATCH -J neurips

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000


conda activate celldreamer

python ../correction_bencmark.py \
--adata_path neurips \
--generated_adata_path /home/icb/alessandro.palma/environment/cfgen/project_folder/celldreamer/experiments_sensitivity/generated_datasets/neurips \
--batch_key Site \
--bio_key cell_type 

# python ../correction_bencmark.py \
# --adata_path c_elegans \
# --generated_adata_path /home/icb/alessandro.palma/environment/cfgen/project_folder/celldreamer/experiments_sensitivity/generated_datasets/c_elegans \
# --batch_key batch \
# --bio_key cell_type 

