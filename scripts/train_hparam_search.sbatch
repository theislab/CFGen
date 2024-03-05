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

python ../celldreamer/train_sccfm.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_dentate hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_encoder_dentategyrus/2d149a86-99da-46cd-804e-58d219c22b5a/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_lung hydra.launcher.array_parallelism=4 hydra.launcher.qos=gpu_normal training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_encoder_lung_atlas/ad7a5a46-8cd8-48b1-aa8a-738e3200cd8a/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_pbmc3k hydra.launcher.array_parallelism=3 trainer.max_epochs=2000 hydra.launcher.qos=gpu_normal training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_encoder_pbmc3k/e7e45b47-75d1-4179-9420-3116a639ce8a/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=neurips hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_neurips hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_encoder training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_encoder_neurips/a1a6770d-151e-446b-9111-c0f0b25ffe94/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=hlca_core hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_encoder_hlca hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_encoder training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_encoder_hlca_core/aff7426c-055c-450f-9f66-256bd5b5a60a/checkpoints/last.ckpt

python ../celldreamer/train_sccfm.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_dentate hydra.launcher.array_parallelism=5 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus/0c0a45f8-ff08-4d26-a92d-7d34f52f649f/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_lung hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal  dataset.encoder_type=learnt_autoencoder training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_lung_atlas/5c06c7fb-ed6c-44e1-a052-a34fb4baa791/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond_pbmc dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_pbmc3k trainer.max_epochs=3000 hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc3k/fadd6319-d854-4da5-b19c-af9e4d16a898/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=neurips hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_neurips hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc3k/fadd6319-d854-4da5-b19c-af9e4d16a898/checkpoints/last.ckpt
python ../celldreamer/train_sccfm.py --multirun dataset=hlca_core hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_hlca hydra.launcher.array_parallelism=3 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc3k/fadd6319-d854-4da5-b19c-af9e4d16a898/checkpoints/last.ckpt


# python ../celldreamer/train.py --multirun dataset=dentategyrus hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_dentate hydra.launcher.array_parallelism=5 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder 
# python ../celldreamer/train.py --multirun dataset=lung_atlas hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_lung hydra.launcher.array_parallelism=2 hydra.launcher.qos=gpu_long dataset.encoder_type=learnt_autoencoder 
# python ../celldreamer/train.py --multirun dataset=pbmc3k hparams_search=fm_resnet_encoder_cond dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=fm_resnet_autoencoder_cond_dentate hydra.launcher.array_parallelism=5 hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder 
