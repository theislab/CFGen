#!/bin/bash

#SBATCH -o ./logs/multilab.out

#SBATCH -e ./logs/multilab.err

#SBATCH -J multilab

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
python ../celldreamer/train_sccfm.py dataset=dentategyrus_multilabel dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_dentategyrus_multilabel trainer.max_epochs=2500 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus/fb452d9f-5618-44c8-afa9-a069494a2219/checkpoints/epoch_59.ckpt
