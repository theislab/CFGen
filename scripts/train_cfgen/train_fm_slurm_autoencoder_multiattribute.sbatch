#!/bin/bash

#SBATCH -o ../logs/cfgen_multiattribute_c.out

#SBATCH -e ../logs/cfgen_multiattribute_c.err

#SBATCH -J cfgen_multiattribute_neurips

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

# NeurIPS
# python ../../cfgen/train_sccfm.py \
# dataset=neurips_multiattribute \
# encoder=default \
# denoising_module=resnet_big \
# launcher=slurm_icb \
# logger.project=cfgen_neurips_final_multiattribute \
# training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_neurips_final/checkpoints/last.ckpt 

# C. Elegans
python ../../cfgen/train_sccfm.py \
dataset=c_elegans_multiattribute \
encoder=encoder_large \
denoising_module=resnet_big \
launcher=slurm_icb \
logger.project=cfgen_hlca_final_multiattribute \
training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_c_elegans_final/checkpoints/last.ckpt 
