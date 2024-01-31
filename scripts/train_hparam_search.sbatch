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

python ../celldreamer/train.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_cond_dentate hydra.launcher.array_parallelism=4 hydra.launcher.qos=gpu_normal
python ../celldreamer/train.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_cond_lung hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal
python ../celldreamer/train.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_cond_pbmc3k_scaled hydra.launcher.array_parallelism=2 hydra.launcher.qos=gpu_long

python ../celldreamer/train.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet denoising_module.model_type=factorized_latent denoising_module.embed_size_factor=False launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_fact_dentate hydra.launcher.array_parallelism=4 hydra.launcher.qos=gpu_normal
python ../celldreamer/train.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet denoising_module.model_type=factorized_latent denoising_module.embed_size_factor=False launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_fact_lung hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal
python ../celldreamer/train.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet denoising_module.model_type=factorized_latent denoising_module.embed_size_factor=False launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_fact_pbmc3k_scaled hydra.launcher.array_parallelism=2 hydra.launcher.qos=gpu_long

# python ../celldreamer/train.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_dentate hydra.launcher.array_parallelism=5 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder 
# python ../celldreamer/train.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_lung hydra.launcher.array_parallelism=2 hydra.launcher.qos=gpu_long dataset.encoder_type=learnt_autoencoder 
# python ../celldreamer/train.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_dentate hydra.launcher.array_parallelism=5 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder 
