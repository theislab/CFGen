#!/bin/bash

#SBATCH -o ./logs/tabula_muris_fm_autoencoder_200.out

#SBATCH -e ./logs/tabula_muris_fm_autoencoder_200.err

#SBATCH -J tabula_muris_fm_autoencoder_200

#SBATCH -p gpu_p

#SBATCH --qos=gpu_normal

#SBATCH --gres=gpu:1

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 2-00:00

#SBATCH --nice=10000

conda activate celldreamer
# python ../celldreamer/train_sccfm.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_dentategyrus training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus/fb452d9f-5618-44c8-afa9-a069494a2219/checkpoints/epoch_59.ckpt
# python ../celldreamer/train_sccfm.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_hlca_core training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_hlca_core/118e6ec2-de99-4b29-9bd7-25ac82c13d06/checkpoints/epoch_199.ckpt
# python ../celldreamer/train_sccfm.py dataset=neurips dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_neurips training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_neurips/e175592b-46a0-4674-baba-db77dd8a079a/checkpoints/epoch_79.ckpt
# python ../celldreamer/train_sccfm.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc3k training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc3k/8b8a9b45-c277-473f-a5be-36282a5e6767/checkpoints/epoch_19.ckpt
# python ../celldreamer/train_sccfm.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc68k training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc68k/c1ee66c2-df1d-4e31-867d-93c8a09a4ebd/checkpoints/epoch_139.ckpt
# python ../celldreamer/train_sccfm.py dataset=c_elegans dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_c_elegans training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_c_elegans/d00adc01-246d-4d0d-92ae-be6d82744cff/checkpoints/epoch_139.ckpt
# python ../celldreamer/train_sccfm.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=10 dataset.one_hot_encode_features=False denoising_module=resnet_small encoder=encoder_small launcher=slurm_icb logger.project=fm_resnet_autoencoder_tabula_muris training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_tabula_muris/60392a17-a375-4b36-8ccc-3a6b7aec5198/checkpoints/epoch_199.ckpt

python ../celldreamer/train_sccfm.py dataset=dentategyrus dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=32 dataset.one_hot_encode_features=False denoising_module=resnet_big encoder=encoder_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_dentategyrus_200 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_dentategyrus_200/962aabdd-b098-4e39-a4cf-baf0932b14e9/checkpoints/epoch_59.ckpt
python ../celldreamer/train_sccfm.py dataset=hlca_core dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=32 dataset.one_hot_encode_features=False denoising_module=resnet_big encoder=encoder_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_hlca_core_200 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_hlca_core_200/5ae24a87-5508-4711-974a-590ab6d749a6/checkpoints/epoch_199.ckpt
python ../celldreamer/train_sccfm.py dataset=neurips dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=32 dataset.one_hot_encode_features=False denoising_module=resnet_big encoder=encoder_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_neurips_200 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_neurips_200/c83bcc09-d9f3-4e22-b6b4-ae4f4fa9d917/checkpoints/epoch_199.ckpt
python ../celldreamer/train_sccfm.py dataset=pbmc3k dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=32 dataset.one_hot_encode_features=False denoising_module=resnet_big encoder=encoder_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc3k_200 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc3k_200/75a432f1-7ee9-483f-82b3-9e0ecaeb25f1/checkpoints/epoch_39.ckpt
python ../celldreamer/train_sccfm.py dataset=pbmc68k dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=32 dataset.one_hot_encode_features=False denoising_module=resnet_big encoder=encoder_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_pbmc68k_200 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_pbmc68k_200/298b0914-4cf2-4905-95df-3efc9fbbad8c/checkpoints/epoch_179.ckpt
python ../celldreamer/train_sccfm.py dataset=c_elegans dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=32 dataset.one_hot_encode_features=False denoising_module=resnet_big encoder=encoder_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_c_elegans_200 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_c_elegans_200/3f074b01-9a87-4b1b-bf71-cd12ecda246f/checkpoints/epoch_199.ckpt
python ../celldreamer/train_sccfm.py dataset=tabula_muris dataset.encoder_type=learnt_autoencoder dataset.cov_embedding_dimensions=32 dataset.one_hot_encode_features=False denoising_module=resnet_big encoder=encoder_big launcher=slurm_icb logger.project=fm_resnet_autoencoder_tabula_muris_200 training_config.encoder_ckpt=/home/icb/alessandro.palma/environment/celldreamer/project_folder/experiments/train_autoencoder_tabula_muris_200/e0a066ae-8429-4010-aecf-a73e78e9d1ea/checkpoints/epoch_199.ckpt
