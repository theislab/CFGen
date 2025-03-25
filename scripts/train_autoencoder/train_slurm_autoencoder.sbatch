#!/bin/bash

#SBATCH -o ../logs/train_autoencoder_c_elegans.out

#SBATCH -e ../logs/train_autoencoder_c_elegans.err

#SBATCH -J train_autoencoder_c_elegans

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

# Dentategyrus
# python ../../cfgen/train_encoder.py dataset=dentategyrus \
# logger.project=train_autoencoder_dentategyrus_final trainer.max_epochs=300

# HLCA
# python ../../cfgen/train_encoder.py dataset=hlca_core encoder=encoder_large \
# logger.project=train_autoencoder_hlca_core_final trainer.max_epochs=300

# HLCA classification
# python ../../cfgen/train_encoder.py dataset=hlca_core_classification \
# logger.project=train_autoencoder_hlca_classification_final trainer.max_epochs=300

# NeurIPS
# python ../../cfgen/train_encoder.py dataset=neurips \
# logger.project=train_autoencoder_neurips_final trainer.max_epochs=300

# PBMC3K
# python ../../cfgen/train_encoder.py dataset=pbmc3k \
# logger.project=train_autoencoder_pbmc3k_final trainer.max_epochs=300

# Tabula Muris
# python ../../cfgen/train_encoder.py dataset=tabula_muris encoder=encoder_large \
# logger.project=train_autoencoder_tabula_muris_final trainer.max_epochs=300

# PBMC covid 
# python ../../cfgen/train_encoder.py dataset=pbmc_covid \
# logger.project=train_autoencoder_pbmc_covid_final trainer.max_epochs=300

# C. Elegans
# python ../../cfgen/train_encoder.py dataset=c_elegans encoder=encoder_large \
# logger.project=train_autoencoder_c_elegans_final trainer.max_epochs=300