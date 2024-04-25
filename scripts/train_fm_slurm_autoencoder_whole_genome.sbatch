#!/bin/bash

#SBATCH -o ./logs/fm_resnet_autoencoder_dentategyrus_whole_genome_no_size_factor.out

#SBATCH -e ./logs/fm_resnet_autoencoder_dentategyrus_whole_genome_no_size_factor.err

#SBATCH -J fm_resnet_autoencoder_dentategyrus_whole_genome_no_size_factor

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=160G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
# python ../celldreamer/train_sccfm.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_dentategyrus_whole_genome training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus_whole_genome/64dcba3b-e80a-4150-8cc8-63562c569088/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_hlca_core_whole_genome training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_hlca_core_whole_genome/2b3472e3-d3f6-4f1b-8199-ceac93f9fcc5/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=neurips dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_neurips_whole_genome training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_neurips_whole_genome/bc40fc96-15c1-4838-bc8d-19d47b79daac/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big  launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc3k_whole_genome training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc3k_whole_genome/a1d7128d-73e2-4c93-b23c-babe015453b8/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc68k_whole_genome training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc68k_whole_genome/49d7fb60-9788-4fa2-9251-651bdfdaa537/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=c_elegans dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_c_elegans_whole_genome training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_c_elegans_whole_genome/23e6fc7f-a7a7-497c-b031-a667e247bd63/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_tabula_muris_whole_genome training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_tabula_muris_whole_genome/20628ff9-cc6f-40a4-aee9-0e22308187a3/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=pbmc_covid dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc_covid_whole_genome training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc_covid/21d30ca0-932d-4891-9796-938e9033b267/checkpoints/last.ckpt

python ../celldreamer/train_sccfm.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb logger.project=fm_resnet_autoencoder_dentategyrus_whole_genome_no_size_factor training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus_whole_genome/64dcba3b-e80a-4150-8cc8-63562c569088/checkpoints/last.ckpt
