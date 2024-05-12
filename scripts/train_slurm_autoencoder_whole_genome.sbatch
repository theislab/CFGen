#!/bin/bash

#SBATCH -o ./logs/pbmc68k_whole_genome.out

#SBATCH -e ./logs/pbmc68k_whole_genome.err

#SBATCH -J pbmc68k_whole_genome

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=160G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
# python ../celldreamer/train_encoder.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder logger.project=off_train_autoencoder_dentategyrus_whole_genome trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large logger.project=off_train_autoencoder_hlca_core_whole_genome trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=neurips dataset.encoder_type=learnt_autoencoder logger.project=off_train_autoencoder_neurips_whole_genome trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder logger.project=off_train_autoencoder_pbmc3k_whole_genome trainer.max_epochs=300 training_config.batch_size=64
# python ../celldreamer/train_encoder.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large logger.project=off_train_autoencoder_pbmc68k_whole_genome trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large logger.project=off_train_autoencoder_tabula_muris_whole_genome trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=c_elegans dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large logger.project=off_train_autoencoder_c_elegans_whole_genome trainer.max_epochs=300

