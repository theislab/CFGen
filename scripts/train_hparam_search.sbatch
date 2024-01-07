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

source $HOME/.bashrc
conda activate celldreamer

python ../celldreamer/train.py --multirun hparams_search=optuna.yaml launcher=slurm_tum logger=hparam_search
python ../celldreamer/train.py --multirun hparams_search=optuna.yaml launcher=slurm_tum logger=hparam_search logger.project=celldreamer_search2
python ../celldreamer/train.py --multirun hparams_search=optuna_resnet denoising_module=resnet launcher=slurm_tum logger=hparam_search logger.project=celldreamer_resnet
python ../celldreamer/train.py --multirun hparams_search=optuna_freeze denoising_module=resnet launcher=slurm_tum logger=hparam_search logger.project=celldreamer_freeze
python ../celldreamer/train.py --multirun hparams_search=optuna_freeze denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=celldreamer_freeze
python ../celldreamer/train.py --multirun hparams_search=optuna_reduced denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=celldreamer_reduced
python ../celldreamer/train.py --multirun hparams_search=optuna_trained denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=celldreamer_trained

python ../celldreamer/train.py --multirun hparams_search=optuna_new_architecture_learnt denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=new_architecture_learnt
python ../celldreamer/train.py --multirun hparams_search=optuna_new_architecture_loggexp denoising_module=resnet launcher=slurm_icb logger=hparam_search logger.project=new_architecture_loggexp