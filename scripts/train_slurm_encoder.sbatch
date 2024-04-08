#!/bin/bash

#SBATCH -o ./logs/pbmc68k_encoder.out

#SBATCH -e ./logs/pbmc68k_encoder.err

#SBATCH -J pbmc68k_encoder

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
# python ../celldreamer/train_encoder.py dataset=dentategyrus dataset.encoder_type=learnt_encoder encoder=encoder logger.project=train_autoencoder_dentategyrus trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=hlca_core dataset.encoder_type=learnt_encoder encoder=encoder logger.project=train_autoencoder_hlca_core trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=neurips dataset.encoder_type=learnt_encoder encoder=encoder logger.project=train_autoencoder_neurips trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=pbmc3k dataset.encoder_type=learnt_encoder encoder=encoder logger.project=train_autoencoder_pbmc3k trainer.max_epochs=200
python ../celldreamer/train_encoder.py dataset=pbmc68k dataset.encoder_type=learnt_encoder encoder=encoder logger.project=train_autoencoder_pbmc68k trainer.max_epochs=200
