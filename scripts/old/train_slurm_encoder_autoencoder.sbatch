#!/bin/bash

#SBATCH -o ./logs/pbmc68k_cfm.out

#SBATCH -e ./logs/pbmc68k_cfm.err

#SBATCH -J pbmc68k_autoencoder

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
python ../celldreamer/train_sccfm.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_dentategyrus training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus/fb452d9f-5618-44c8-afa9-a069494a2219/checkpoints/epoch_59.ckpt


# python ../celldreamer/train_encoder.py dataset=dentategyrus dataset.encoder_type=learnt_encoder logger.project=train_encoder_dentategyrus
# python ../celldreamer/train_encoder.py dataset=lung_atlas dataset.encoder_type=learnt_encoder logger.project=train_encoder_lung_atlas
# python ../celldreamer/train_encoder.py dataset=pbmc3k dataset.encoder_type=learnt_encoder logger.project=train_encoder_pbmc3k
# python ../celldreamer/train_encoder.py dataset=pbmc68k dataset.encoder_type=learnt_encoder logger.project=train_encoder_pbmc68k

# python ../celldreamer/train_encoder.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_dentategyrus trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=neurips dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_neurips trainer.max_epochs=300


# python ../celldreamer/train_encoder.py dataset=lung_atlas dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_lung_atlas
# python ../celldreamer/train_encoder.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_pbmc3k
# python ../celldreamer/train_encoder.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_pbmc68k
# python ../celldreamer/train_encoder.py dataset=neurips dataset.encoder_type=learnt_autoencoder logger.project=train_autoencoder_neurips trainer.max_epochs=300
# python ../celldreamer/train_encoder.py dataset=hlca_core dataset.encoder_type=learnt_encoder logger.project=train_encoder_hlca_core trainer.max_epochs=300


# python ../celldreamer/train_encoder.py dataset=lung_atlas encoder.covariate_specific_theta=True dataset.conditioning_covariate=batch dataset.encoder_type=learnt_encoder logger.project=train_encoder_dentategyrus_cond
