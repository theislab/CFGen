#!/bin/bash

#SBATCH -o ./logs/train_slurm_hp2.out

#SBATCH -e ./logs/train_slurm_hp2.err

#SBATCH -J train_slurm_hp

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

source $HOME/.bashrc
conda activate celldreamer

# python ../celldreamer/train.py --multirun dataset=dentategyrus hparams_search=fm_mlp_encoder_cond denoising_module=default launcher=slurm_icb logger=hparam_search logger.project=fm_mlp_encoder_cond_dentate hydra.launcher.array_parallelism=3 
# python ../celldreamer/train.py --multirun dataset=dentategyrus hparams_search=fm_mlp_encoder_cond denoising_module=default launcher=slurm_icb logger=hparam_search logger.project=fm_mlp_autoencoder_cond_dentate_small hydra.launcher.array_parallelism=2 dataset.encoder_type=learnt_autoencoder 
# python ../celldreamer/train.py --multirun dataset=lung_atlas hparams_search=fm_mlp_encoder_cond denoising_module=default launcher=slurm_icb logger=hparam_search logger.project=fm_mlp_autoencoder_cond_lung_small hydra.launcher.array_parallelism=2 dataset.encoder_type=learnt_autoencoder 

python ../celldreamer/train.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_dentate hydra.launcher.array_parallelism=4 hydra.launcher.qos=gpu_normal
python ../celldreamer/train.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_lung hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal
python ../celldreamer/train.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_pbmc3k_scaled hydra.launcher.array_parallelism=2 hydra.launcher.qos=gpu_normal

python ../celldreamer/train_sccfm.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_dentate hydra.launcher.array_parallelism=4 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder generative_model.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus/0c0a45f8-ff08-4d26-a92d-7d34f52f649f/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_lung hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal  dataset.encoder_type=learnt_autoencoder generative_model.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_lung_atlas/5c06c7fb-ed6c-44e1-a052-a34fb4baa791/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_pbmc3k_scaled hydra.launcher.array_parallelism=2 hydra.launcher.qos=gpu_normal  dataset.encoder_type=learnt_autoencoder generative_model.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc3k/fadd6319-d854-4da5-b19c-af9e4d16a898/checkpoints/last.ckpt

# python ../celldreamer/train.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_dentate hydra.launcher.array_parallelism=5 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder 
# python ../celldreamer/train.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_lung hydra.launcher.array_parallelism=2 hydra.launcher.qos=gpu_long dataset.encoder_type=learnt_autoencoder 
# python ../celldreamer/train.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_dentate hydra.launcher.array_parallelism=5 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder 
