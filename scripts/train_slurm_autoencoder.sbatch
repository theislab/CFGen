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

conda activate celldreamer
# python ../celldreamer/train_encoder.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_dentategyrus trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_hlca_core trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=neurips dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_neurips trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_pbmc3k trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_pbmc68k trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_tabula_muris trainer.max_epochs=200
# python ../celldreamer/train_encoder.py dataset=c_elegans dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_c_elegans trainer.max_epochs=200
python ../celldreamer/train_encoder.py dataset=pbmc_covid dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_pbmc_covid trainer.max_epochs=300

# python ../celldreamer/train_encoder.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_dentategyrus_200 trainer.max_epochs=200 
# python ../celldreamer/train_encoder.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_hlca_core_200 trainer.max_epochs=200 
# python ../celldreamer/train_encoder.py dataset=neurips dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_neurips_200 trainer.max_epochs=200 
# python ../celldreamer/train_encoder.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_pbmc3k_200 trainer.max_epochs=200 
# python ../celldreamer/train_encoder.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_pbmc68k_200 trainer.max_epochs=200 
# python ../celldreamer/train_encoder.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_tabula_muris_200 trainer.max_epochs=200 
# python ../celldreamer/train_encoder.py dataset=c_elegans dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_c_elegans_200 trainer.max_epochs=200