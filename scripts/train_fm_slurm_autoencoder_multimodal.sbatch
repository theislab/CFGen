#!/bin/bash

#SBATCH -o ./logs/pbmc10k_multi_fm_joint.out

#SBATCH -e ./logs/pbmc10k_multi_fm_joint.err

#SBATCH -J pbmc10k_multi_fm_joint

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --exclude=gpusrv26

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

# python ../celldreamer/train_sccfm.py dataset=neurips_multimodal dataset.encoder_type=learnt_autoencoder \
# dataset.one_hot_encode_features=False denoising_module=resnet_big \
# denoising_module.multimodal=True denoising_module.is_binarized=False encoder=encoder_multimodal_joint \
# encoder.multimodal=True encoder.is_binarized=False trainer.max_epochs=2000 launcher=slurm_icb \
# logger.project=fm_resnet_autoencoder_neurips_multimodal_joint \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_neurips_multimodal_joint/1c26a129-9280-421c-b941-1203cd3887a2/checkpoints/last.ckpt

# python ../celldreamer/train_sccfm.py dataset=neurips_multimodal_binary dataset.encoder_type=learnt_autoencoder \
# dataset.one_hot_encode_features=False denoising_module=resnet_big \
# denoising_module.multimodal=True denoising_module.is_binarized=True encoder=encoder_multimodal_joint \
# encoder.multimodal=True encoder.is_binarized=True trainer.max_epochs=1000 launcher=slurm_icb \
# logger.project=REBUTTAL_fm_resnet_autoencoder_neurips_multimodal_bin_joint \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_neurips_multimodal_joint_binary/2a7f0851-371c-489b-9a3b-cadd7db7a022/checkpoints/epoch_19.ckpt

python ../celldreamer/train_sccfm.py dataset=pbmc10k_multimodal dataset.encoder_type=learnt_autoencoder \
dataset.one_hot_encode_features=False denoising_module=resnet_big \
denoising_module.multimodal=True denoising_module.is_binarized=True encoder=encoder_multimodal_joint \
encoder.multimodal=True encoder.is_binarized=True trainer.max_epochs=2000 launcher=slurm_icb \
logger.project=REBUTTAL_fm_resnet_autoencoder_pbmc10k_multimodal_joint \
training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/cfgen/project_folder/experiments/REBUTTAL_train_autoencoder_pbmc10k_multimodal_joint/e00c2685-94b1-4a0a-b4a0-8e1fef5b8336/checkpoints/epoch_39.ckpt


# python ../celldreamer/train_sccfm.py dataset=neurips_multimodal dataset.encoder_type=learnt_autoencoder \
# dataset.one_hot_encode_features=False denoising_module=resnet_big \
# denoising_module.multimodal=True denoising_module.is_binarized=False encoder=encoder_multimodal \
# encoder.multimodal=True encoder.is_binarized=False trainer.max_epochs=1500 launcher=slurm_icb \
# logger.project=fm_resnet_autoencoder_neurips_multimodal \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_neurips_multimodal/9e32dc3f-46f5-4eca-849c-40072c9ef0b0/checkpoints/last.ckpt

# python ../celldreamer/train_sccfm.py dataset=pbmc10k_multimodal dataset.encoder_type=learnt_autoencoder \
# dataset.one_hot_encode_features=False denoising_module=resnet_big \
# denoising_module.multimodal=True denoising_module.is_binarized=True encoder=encoder_multimodal \
# encoder.multimodal=True encoder.is_binarized=True trainer.max_epochs=1500 launcher=slurm_icb \
# logger.project=fm_resnet_autoencoder_pbmc10k_multimodal \
# training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc10k_multimodal/17b54887-53a4-4983-8afc-7fdc9d3274c8/checkpoints/last.ckpt
