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


python ../celldreamer/train.py denoising_module=default launcher=slurm_icb generative_model.pretraining_encoder_epochs=200 trainer.max_epochs=1500 dataset=dentategyrus dataset.encoder_type=learnt_encoder denoising_module.conditional=True training_config.batch_size=256 logger.project=fm_mlp_encoder_cond
# python ../celldreamer/train.py denoising_module=default launcher=slurm_icb generative_model.pretraining_encoder_epochs=5 trainer.max_epochs=7 dataset=dentategyrus dataset.encoder_type=learnt_encoder denoising_module.conditional=True training_config.batch_size=256 logger.offline=True
# python ../celldreamer/train.py denoising_module=resnet launcher=slurm_icb dataset.encoder_type=learnt
# python ../celldreamer/train.py denoising_module=resnet launcher=slurm_icb dataset.encoder_type=learnt generative_model.pretraining_encoder_epochs=100 trainer.max_epochs=100 generative_model.learning_rate=0.001
