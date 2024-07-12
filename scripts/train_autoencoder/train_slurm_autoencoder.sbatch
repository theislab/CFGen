#!/bin/bash

#SBATCH -o ./logs/pbmc_covid.out

#SBATCH -e ./logs/pbmc_covid.err

#SBATCH -J pbmc_covid

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

# Dentategyrus
python ../cfgen/train_encoder.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder \
logger.project=train_autoencoder_dentategyrus trainer.max_epochs=300

# Hlca_core
python ../cfgen/train_encoder.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder \
logger.project=train_autoencoder_hlca_core trainer.max_epochs=300

# NeurIPS
python ../cfgen/train_encoder.py dataset=neurips dataset.encoder_type=learnt_autoencoder \
logger.project=train_autoencoder_neurips trainer.max_epochs=300

# pbmc3k
python ../cfgen/train_encoder.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder \
logger.project=train_autoencoder_pbmc3k trainer.max_epochs=300

# tabula_muris
python ../cfgen/train_encoder.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder \
logger.project=train_autoencoder_tabula_muris trainer.max_epochs=300

# pbmc covid
python ../cfgen/train_encoder.py dataset=pbmc_covid dataset.encoder_type=learnt_autoencoder \
logger.project=train_autoencoder_pbmc_covid trainer.max_epochs=300

