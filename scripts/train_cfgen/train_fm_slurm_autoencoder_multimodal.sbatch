#!/bin/bash

#SBATCH -o ../logs/cfgen_multimodal.out

#SBATCH -e ../logs/cfgen_multimodal.err

#SBATCH -J cfgen_multimodal

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

# Train autoencoder with joint latent space  
# python ../../cfgen/train_sccfm.py dataset=pbmc10k_multimodal dataset.encoder_type=learnt_autoencoder \
# dataset.one_hot_encode_features=False denoising_module=resnet_big \
# denoising_module.is_binarized=True encoder=encoder_multimodal_joint \
# encoder.is_binarized=True trainer.max_epochs=1000 launcher=slurm_icb \
# logger.project=fm_resnet_autoencoder_pbmc10k_multimodal_joint \
# training_config.encoder_ckpt=/path/to/checkpoints/

# # Train autoencoder with separate latent space   
# python ../../cfgen/train_sccfm.py dataset=pbmc10k_multimodal dataset.encoder_type=learnt_autoencoder \
# dataset.one_hot_encode_features=False denoising_module=resnet_big \
# denoising_module.is_binarized=True encoder=encoder_multimodal \
# encoder.is_binarized=True trainer.max_epochs=1000 launcher=slurm_icb \
# logger.project=fm_resnet_autoencoder_pbmc10k_multimodal \
# training_config.encoder_ckpt=/path/to/checkpoints/
