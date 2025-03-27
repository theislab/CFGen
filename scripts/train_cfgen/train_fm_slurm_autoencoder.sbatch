#!/bin/bash

#SBATCH -o ../logs/cfgen_tabula_muris.out

#SBATCH -e ../logs/cfgen_tabula_muris.err

#SBATCH -J cfgen_tabula_muris

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

# Dentategyrus
# python ../../cfgen/train_sccfm.py \
# dataset=dentategyrus \
# encoder=default \
# dataset.one_hot_encode_features=False \
# denoising_module=resnet_small \
# denoising_module.embed_size_factor=True \
# launcher=slurm_icb \
# logger.project=cfgen_dentategyrus_final \
# training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_dentategyrus_final/checkpoints/last.ckpt

# Hlca core
# python ../../cfgen/train_sccfm.py \
# dataset=hlca_core \
# encoder=encoder_large \
# dataset.one_hot_encode_features=False \
# denoising_module=resnet_big \
# launcher=slurm_icb \
# logger.project=cfgen_hlca_final \
# training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_hlca_core_final/checkpoints/last.ckpt

# Hlca classification
python ../../cfgen/train_sccfm.py \
dataset=hlca_core_classification \
encoder=default \
dataset.one_hot_encode_features=False \
denoising_module=resnet_big \
launcher=slurm_icb \
logger.project=cfgen_hlca_final \
training_config.encoder_ckpt=project_folder/experiments/autoencoder_ckpt/train_autoencoder_hlca_classification_final/checkpoints/last.ckpt

# NeurIPS
# python ../../cfgen/train_sccfm.py \
# dataset=neurips \
# encoder=default \
# dataset.one_hot_encode_features=False \
# denoising_module=resnet_big \
# launcher=slurm_icb \
# logger.project=cfgen_neurips_final \
# training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_neurips_final/checkpoints/last.ckpt

# pbmc3k
# python ../../cfgen/train_sccfm.py \
# dataset=pbmc3k \
# encoder=default \
# dataset.one_hot_encode_features=False \
# denoising_module=resnet_small \
# launcher=slurm_icb \
# logger.project=cfgen_pbmc3k_final \
# training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_pbmc3k_final/checkpoints/last.ckpt \
# training_config.batch_size=64

# # tabula_muris
# python ../../cfgen/train_sccfm.py \
# dataset=tabula_muris \
# encoder=encoder_large \
# dataset.one_hot_encode_features=False \
# denoising_module=resnet_big \
# launcher=slurm_icb \
# logger.project=cfgen_tabula_muris_final \
# training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_tabula_muris_final/checkpoints/last.ckpt

# pbmc covid
# python ../../cfgen/train_sccfm.py \
# dataset=pbmc_covid \
# encoder=default \
# dataset.one_hot_encode_features=False \
# denoising_module=resnet_big \
# launcher=slurm_icb \
# logger.project=cfgen_pbmc_covid_final \
# training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_pbmc_covid_final/checkpoints/last.ckpt

# C. Elegans
python ../../cfgen/train_sccfm.py \
dataset=c_elegans \
encoder=encoder_large \
dataset.one_hot_encode_features=False \
denoising_module=resnet_big \
launcher=slurm_icb \
logger.project=cfgen_hlca_final \
training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_c_elegans_final/last.ckpt
