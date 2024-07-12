#!/bin/bash

#SBATCH -o ../logs/cfgen_scrnaseq.out

#SBATCH -e ../logs/cfgen_scrnaseq.err

#SBATCH -J cfgen_scrnaseq

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

# Dentategyrus
python ../celldreamer/train_sccfm.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder encoder=autoencoder_small \
dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb \
logger.project=off_fm_resnet_autoencoder_dentategyrus_whole_genome_no_size_factor \
training_config.encoder_ckpt=/path/to/checkpoints/

# Hlca_core
python ../celldreamer/train_sccfm.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb \
logger.project=off_fm_resnet_autoencoder_hlca_core_whole_genome \
training_config.encoder_ckpt=/path/to/checkpoints/

# NeurIPS
python ../celldreamer/train_sccfm.py dataset=neurips dataset.encoder_type=learnt_autoencoder encoder=autoencoder_small \
dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb \
logger.project=off_fm_resnet_autoencoder_neurips_whole_genome \
training_config.encoder_ckpt=/path/to/checkpoints/

# pbmc3k
python ../celldreamer/train_sccfm.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder encoder=autoencoder_small \
dataset.one_hot_encode_features=False denoising_module=resnet_big  \
launcher=slurm_icb logger.project=off_fm_resnet_autoencoder_pbmc3k_whole_genome \
training_config.encoder_ckpt=/path/to/checkpoints training_config.batch_size=64

# tabula_muris
python ../celldreamer/train_sccfm.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
dataset.one_hot_encode_features=False denoising_module=resnet_big \
launcher=slurm_icb logger.project=off_fm_resnet_autoencoder_tabula_muris_whole_genome \
training_config.encoder_ckpt=/path/to/checkpoints/

# pbmc covid
python ../celldreamer/train_sccfm.py dataset=pbmc_covid dataset.encoder_type=learnt_autoencoder encoder=encoder_small \
dataset.one_hot_encode_features=False denoising_module=resnet_big \
launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc_covid_whole_genome \
training_config.encoder_ckpt=/path/to/checkpoints/
