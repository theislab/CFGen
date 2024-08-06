#!/bin/bash

#SBATCH -o ./logs/joint.out

#SBATCH -e ./logs/joint.err

#SBATCH -J joint

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
# python ../celldreamer/train_encoder.py dataset=neurips_multimodal dataset.encoder_type=learnt_autoencoder encoder=encoder_multimodal encoder.is_binarized=False logger.project=train_autoencoder_neurips_multimodal trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=pbmc10k_multimodal dataset.encoder_type=learnt_autoencoder encoder=encoder_multimodal encoder.is_binarized=True logger.project=train_autoencoder_pbmc10k_multimodal trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=neurips_multimodal dataset.encoder_type=learnt_autoencoder encoder=encoder_multimodal_joint encoder.is_binarized=False logger.project=train_autoencoder_neurips_multimodal_joint trainer.max_epochs=300
python ../celldreamer/train_encoder.py dataset=pbmc10k_multimodal dataset.encoder_type=learnt_autoencoder encoder=encoder_multimodal_joint encoder.is_binarized=True logger.project=REBUTTAL_train_autoencoder_pbmc10k_multimodal_joint trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=neurips_multimodal_binary dataset.encoder_type=learnt_autoencoder encoder=encoder_multimodal encoder.is_binarized=True logger.project=train_autoencoder_neurips_multimodal_binary trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=neurips_multimodal_binary dataset.encoder_type=learnt_autoencoder encoder=encoder_multimodal_joint encoder.is_binarized=True logger.project=train_autoencoder_neurips_multimodal_joint_binary trainer.max_epochs=300

