#!/bin/bash

#SBATCH -o ./logs/tabula_muris.out

#SBATCH -e ./logs/tabula_muris.err

#SBATCH -J tabula_muris

#SBATCH --exclude=supergpu02

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=160G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
# python ../celldreamer/train_sccfm.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_dentategyrus_whole_genome_128 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/train_autoencoder_dentategyrus_whole_genome/64dcba3b-e80a-4150-8cc8-63562c569088/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_hlca_core_whole_genome_128 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/train_autoencoder_hlca_core_whole_genome/2b3472e3-d3f6-4f1b-8199-ceac93f9fcc5/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=neurips dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_neurips_whole_genome_128 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/train_autoencoder_neurips_whole_genome/bc40fc96-15c1-4838-bc8d-19d47b79daac/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big  launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc3k_whole_genome_128 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/train_autoencoder_pbmc3k_whole_genome/a1d7128d-73e2-4c93-b23c-babe015453b8/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc68k_whole_genome_128 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/train_autoencoder_pbmc68k_whole_genome/49d7fb60-9788-4fa2-9251-651bdfdaa537/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=c_elegans dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_c_elegans_whole_genome_128 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/train_autoencoder_c_elegans_whole_genome/23e6fc7f-a7a7-497c-b031-a667e247bd63/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_tabula_muris_whole_genome_128 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/train_autoencoder_tabula_muris_whole_genome/20628ff9-cc6f-40a4-aee9-0e22308187a3/checkpoints/last.ckpt
# python ../celldreamer/train_sccfm.py dataset=pbmc_covid dataset.encoder_type=learnt_autoencoder encoder=encoder_small dataset.one_hot_encode_features=False denoising_module=resnet_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc_covid_whole_genome_128 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/train_autoencoder_pbmc_covid/21d30ca0-932d-4891-9796-938e9033b267/checkpoints/last.ckpt

# # Dentategyrus
# python ../cfgen/train_sccfm.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder encoder=encoder_small \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
# logger.project=off_fm_resnet_autoencoder_dentategyrus_whole_genome_no_size_factor \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_dentategyrus_whole_genome/feab789d-5b04-45c4-b26a-cbaf0d44755a/checkpoints/epoch_59.ckpt

# # Hlca_core
# python ../cfgen/train_sccfm.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
# logger.project=off_fm_resnet_autoencoder_hlca_core_whole_genome \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_hlca_core_whole_genome/80ee94e4-ba9f-4481-a2db-8fa005a5cf67/checkpoints/epoch_99.ckpt

# # NeurIPS
# python ../cfgen/train_sccfm.py dataset=neurips dataset.encoder_type=learnt_autoencoder encoder=encoder_small \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
# logger.project=off_fm_resnet_autoencoder_neurips_whole_genome \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_neurips_whole_genome/eabc6534-947b-4486-8012-c9e351b297ca/checkpoints/epoch_59.ckpt

# # pbmc3k
# python ../cfgen/train_sccfm.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder encoder=encoder_small \
# dataset.one_hot_encode_features=False denoising_module=resnet_big  denoising_module.embed_size_factor=False  \
# launcher=slurm_icb logger.project=off_fm_resnet_autoencoder_pbmc3k_whole_genome \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_pbmc3k_whole_genome/3d875b67-d126-496c-9860-38ff7f25d7c4/checkpoints/epoch_39.ckpt \
# training_config.batch_size=64

# # pbmc68k
# python ../cfgen/train_sccfm.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False \
# launcher=slurm_icb logger.project=off_fm_resnet_autoencoder_pbmc68k_whole_genome \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_pbmc68k_whole_genome/97f9b5ed-0ed4-4149-a6cd-2d06195eba61/checkpoints/epoch_19.ckpt

# # tabula_muris
# python ../cfgen/train_sccfm.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder \
# encoder=autoencoder_large dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False \
# launcher=slurm_icb logger.project=off_fm_resnet_autoencoder_tabula_muris_whole_genome \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_tabula_muris_whole_genome/df8d1d1a-085d-403c-8323-beb4c5a3f4cd/checkpoints/epoch_79.ckpt

# # # c_elegans
# python ../cfgen/train_sccfm.py dataset=c_elegans dataset.encoder_type=learnt_autoencoder \
# encoder=autoencoder_large dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False \
# launcher=slurm_icb logger.project=off_fm_resnet_autoencoder_c_elegans_whole_genome \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_c_elegans_whole_genome/bd5c1915-ea4d-4428-9aeb-99d3346bcc55/checkpoints/epoch_19.ckpt

# # HLCA CORE - CLASSIFICATION
# python ../cfgen/train_sccfm.py dataset=hlca_core_classification dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
# logger.project=CLASSIFIER_off_fm_resnet_autoencoder_hlca_core_whole_genome \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/CLASSIFICATION_off_train_autoencoder_hlca_core/3dde9d9c-2fbc-4417-bdd7-9e8ebe1c948b/checkpoints/last.ckpt

# # HLCA CORE - CLASSIFICATION
# python ../cfgen/train_sccfm.py dataset=pbmc10k_multimodal_rna dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
# dataset.one_hot_encode_features=False denoising_module=resnet_small denoising_module.embed_size_factor=False launcher=slurm_icb \
# logger.project=REBUTTALS_PBMC10K_FM \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/REBUTTALS_PBMC10K/201a2a6e-9dd8-4641-a62b-fc1cdf874724/checkpoints/last.ckpt

# Kang - CLASSIFICATION
# python ../cfgen/train_sccfm.py dataset=kang dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
# dataset.one_hot_encode_features=False denoising_module=resnet_small denoising_module.embed_size_factor=False launcher=slurm_icb \
# logger.project=REBUTTALS_KANG_FM \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/REBUTTALS_KANG_AUTOENCODER/dfbcc385-d315-4f5a-97ad-9f29c14c80a8/checkpoints/epoch_19.ckpt