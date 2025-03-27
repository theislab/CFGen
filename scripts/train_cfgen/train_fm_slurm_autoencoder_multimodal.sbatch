#!/bin/bash

#SBATCH -o ../logs/cfgen_multimodal_joint.out

#SBATCH -e ../logs/cfgen_multimodal_joint.err

#SBATCH -J cfgen_multimodal_joint

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

# Train autoencoder with joint latent space  
python ../../cfgen/train_sccfm.py \
dataset=pbmc10k_multimodal \
dataset.one_hot_encode_features=False \
denoising_module=resnet_big \
denoising_module.is_binarized=True \
encoder=encoder_multimodal_joint \
encoder.is_binarized=True \
launcher=slurm_icb \
logger.project=fm_pbmc10k_multimodal_joint \
training_config.encoder_ckpt=../../project_folder/experiments/autoencoder_ckpt/train_autoencoder_pbmc10k_multimodal_joint/checkpoints/last.ckpt