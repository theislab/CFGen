#!/bin/bash

#SBATCH -o ./logs/train_slurm.out

#SBATCH -e ./logs/train_slurm.err

#SBATCH -J train_slurm

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer

python ../celldreamer/train_sccfm.py dataset=dentategyrus dataset.one_hot_encode_features=False denoising_module=resnet launcher=slurm_icb logger.project=fm_resnet_autoencoder_dentate hydra.launcher.qos=gpu_normal dataset.encoder_type=learnt_autoencoder training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus/0c0a45f8-ff08-4d26-a92d-7d34f52f649f/checkpoints/last.ckpt

# python ../celldreamer/train.py denoising_module=default launcher=slurm_icb generative_model.pretraining_encoder_epochs=200 trainer.max_epochs=1500 dataset=dentategyrus dataset.encoder_type=learnt_encoder denoising_module.conditional=True training_config.batch_size=256 logger.project=fm_mlp_encoder_cond
# python ../celldreamer/train.py denoising_module=default launcher=slurm_icb generative_model.pretraining_encoder_epochs=5 trainer.max_epochs=7 dataset=dentategyrus dataset.encoder_type=learnt_encoder denoising_module.conditional=True training_config.batch_size=256 logger.offline=True
# python ../celldreamer/train.py denoising_module=resnet launcher=slurm_icb dataset.encoder_type=learnt
# python ../celldreamer/train.py denoising_module=resnet launcher=slurm_icb dataset.encoder_type=learnt generative_model.pretraining_encoder_epochs=100 trainer.max_epochs=100 generative_model.learning_rate=0.001
