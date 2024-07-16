#!/bin/bash

#SBATCH -o ../logs/train_autoencoder.out

#SBATCH -e ../logs/train_autoencoder.err

#SBATCH -J train_autoencoder

#SBATCH --exclude=supergpu02

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

# Dentategyrus
python ../../cfgen/train_encoder.py dataset=dentategyrus \
logger.project=train_autoencoder_dentategyrus trainer.max_epochs=300

# # Hlca_core
# python ../../cfgen/train_encoder.py dataset=hlca_core \
# logger.project=train_autoencoder_hlca_core trainer.max_epochs=300

# # NeurIPS
# python ../../cfgen/train_encoder.py dataset=neurips \
# logger.project=train_autoencoder_neurips trainer.max_epochs=300

# # pbmc3k
# python ../../cfgen/train_encoder.py dataset=pbmc3k \
# logger.project=train_autoencoder_pbmc3k trainer.max_epochs=300

# # tabula_muris
# python ../../cfgen/train_encoder.py dataset=tabula_muris \
# logger.project=train_autoencoder_tabula_muris trainer.max_epochs=300

# # pbmc covid
# python ../../cfgen/train_encoder.py dataset=pbmc_covid \
# logger.project=train_autoencoder_pbmc_covid trainer.max_epochs=300
