#!/bin/bash

#SBATCH -o ./logs/pbmc10k_multi.out

#SBATCH -e ./logs/pbmc10k_multi.err

#SBATCH -J pbmc10k_multi

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
# python ../celldreamer/train_encoder.py dataset=neurips_multimodal dataset.encoder_type=learnt_autoencoder encoder=encoder_multimodal logger.project=train_autoencoder_neurips_multimodal trainer.max_epochs=200
python ../celldreamer/train_encoder.py dataset=pbmc10k_multimodal dataset.encoder_type=learnt_autoencoder encoder=encoder_multimodal encoder.is_binarized=True logger.project=train_autoencoder_pbmc10k_multimodal trainer.max_epochs=200
