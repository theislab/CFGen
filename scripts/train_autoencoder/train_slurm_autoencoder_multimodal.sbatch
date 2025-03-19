#!/bin/bash

#SBATCH -o ../logs/train_autoencoder_multimodal_joint.out

#SBATCH -e ../logs/train_autoencoder_multimodal_joint.err

#SBATCH -J train_autoencoder_multimodal_joint

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

# PBMC10K multimodal
# python ../../cfgen/train_encoder.py dataset=pbmc10k_multimodal encoder=encoder_multimodal \
# encoder.is_binarized=True logger.project=train_autoencoder_pbmc10k trainer.max_epochs=300

# PBMC10K multimodal - joint latent space
python ../../cfgen/train_encoder.py dataset=pbmc10k_multimodal encoder=encoder_multimodal_joint \
encoder.is_binarized=True logger.project=train_autoencoder_pbmc10k_multimodal_joint trainer.max_epochs=300