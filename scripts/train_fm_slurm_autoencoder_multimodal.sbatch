#!/bin/bash

#SBATCH -o ./logs/pbmc10k_multi_fm.out

#SBATCH -e ./logs/pbmc10k_multi_fm.err

#SBATCH -J pbmc10k_multi_fm

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

python ../celldreamer/train_sccfm.py dataset=neurips_multimodal dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=32 dataset.one_hot_encode_features=False denoising_module=resnet_big denoising_module.multimodal=True denoising_module.is_binarized=False encoder=encoder_multimodal encoder.multimodal=True encoder.is_binarized=False trainer.max_epochs=1500 launcher=slurm_icb logger.project=fm_resnet_autoencoder_neurips_multimodal training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_neurips_multimodal/fb44592a-78b7-4210-adaa-bd5df7665be2/checkpoints/last.ckpt
