#!/bin/bash

#SBATCH -o ./logs/UNGUIDED_MULTILAB_TABULA.out

#SBATCH -e ./logs/UNGUIDED_MULTILAB_TABULA.err

#SBATCH -J UNGUIDED_MULTILAB_TABULA

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
export HYDRA_FULL_ERROR=1

# python ../celldreamer/train_sccfm.py dataset=dentategyrus_multilabel dataset.encoder_type=learnt_autoencoder encoder=encoder_small \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
# denoising_module.guided_conditioning=False logger.project=UNGUIDED_MULTILAB_DENTATE trainer.max_epochs=2000 \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_dentategyrus_whole_genome/feab789d-5b04-45c4-b26a-cbaf0d44755a/checkpoints/epoch_59.ckpt

# python ../celldreamer/train_sccfm.py dataset=c_elegans_multilabel dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
# denoising_module.guided_conditioning=False logger.project=UNGUIDED_MULTILAB_C_ELEGANS trainer.max_epochs=2000 \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_c_elegans_whole_genome/bd5c1915-ea4d-4428-9aeb-99d3346bcc55/checkpoints/epoch_19.ckpt

# python ../celldreamer/train_sccfm.py dataset=neurips_multilabel dataset.encoder_type=learnt_autoencoder encoder=encoder_small \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
# denoising_module.guided_conditioning=False logger.project=UNGUIDED_MULTILAB_NEURIPS trainer.max_epochs=2000 \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_neurips_whole_genome/eabc6534-947b-4486-8012-c9e351b297ca/checkpoints/epoch_59.ckpt

# python ../celldreamer/train_sccfm.py dataset=neurips_multilabel_site dataset.encoder_type=learnt_autoencoder encoder=encoder_small \
# dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
# denoising_module.guided_conditioning=False logger.project=UNGUIDED_MULTILAB_NEURIPS_SITE trainer.max_epochs=2000 \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_neurips_whole_genome/eabc6534-947b-4486-8012-c9e351b297ca/checkpoints/epoch_59.ckpt

python ../celldreamer/train_sccfm.py dataset=tabula_muris_multilabel dataset.encoder_type=learnt_autoencoder encoder=autoencoder_large \
dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.embed_size_factor=False launcher=slurm_icb \
denoising_module.guided_conditioning=False logger.project=UNGUIDED_MULTILAB_TABULA trainer.max_epochs=2000 \
training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/off_train_autoencoder_tabula_muris_whole_genome/df8d1d1a-085d-403c-8323-beb4c5a3f4cd/checkpoints/epoch_79.ckpt
