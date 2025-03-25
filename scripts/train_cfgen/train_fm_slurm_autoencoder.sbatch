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
# python ../../cfgen/train_sccfm.py dataset=dentategyrus encoder=default \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=True launcher=slurm_icb \
# logger.project=cfgen_dentategyrus training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/off_train_autoencoder_dentategyrus_whole_genome/feab789d-5b04-45c4-b26a-cbaf0d44755a/checkpoints/epoch_59.ckpt

# # Hlca_core
# python ../../cfgen/train_sccfm.py dataset=hlca_core encoder=autoencoder_large \
# dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb \
# logger.project=cfgen_hlca training_config.encoder_ckpt=/path/to/checkpoints/

# # NeurIPS
# python ../../cfgen/train_sccfm.py dataset=neurips encoder=default \
# dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb \
# logger.project=cfgen_neurips training_config.encoder_ckpt=/path/to/checkpoints/

# # pbmc3k
# python ../../cfgen/train_sccfm.py dataset=pbmc3k encoder=default \
# dataset.one_hot_encode_features=False denoising_module=resnet_big  \
# launcher=slurm_icb logger.project=cfgen_pbmc3k training_config.encoder_ckpt=/path/to/checkpoints training_config.batch_size=64

# # tabula_muris
# python ../../cfgen/train_sccfm.py dataset=tabula_muris encoder=autoencoder_large \
# dataset.one_hot_encode_features=False denoising_module=resnet_big \
# launcher=slurm_icb logger.project=cfgen_tabula_muris training_config.encoder_ckpt=/path/to/checkpoints/

# # pbmc covid
# python ../../cfgen/train_sccfm.py dataset=pbmc_covid encoder=default \
# dataset.one_hot_encode_features=False denoising_module=resnet_big \
# launcher=slurm_icb logger.project=cfgen_pbmc_covid training_config.encoder_ckpt=/path/to/checkpoints/
