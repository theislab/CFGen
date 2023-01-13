import argparse
import sys
import yaml
from pathlib import Path
from math import ceil
import os
import torch
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.utilities.model_summary import ModelSummary

root = os.path.dirname(os.path.abspath(os.curdir))
sys.path.append(root)

from scgm.models.generative_models.autoencoder import AutoEncoder
from scgm.data.utils import get_estimator, get_model_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--loss_func', type=str, default='MSE')  # To Do: think about different loss functions - CB, MSE
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_norm', default=True, type=lambda x: x.lower() in ['true', '1', '1.'])
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--gain_weight_init_first_layer', default=4., type=float)
    parser.add_argument('--gain_weight_init', default=0.75, type=float)
    # Dataloading parameters
    parser.add_argument('--RETRIEVAL_BATCH_SIZE', type=int, default=35000)
    parser.add_argument('--SHUFFLE_BUFFER_SIZE', type=int, default=15000)
    # Checkpointing
    parser.add_argument('--version', type=str, default='')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    # GET GPU AND ARGS
    if torch.cuda.is_available():
        print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    args = parse_args()

    # GET HYPERPARAMETER
    if args.model == 'mlp':
        hparams = yaml.safe_load(Path(root + '/scgm/models/generative_models/hparams_ae.yaml').read_text())
        model_class = AutoEncoder  # maybe we need better variable names...
    else:
        raise ValueError('Only MLP is implemented yet, TO DO: implement TabNet')

    # FIX SEED FOR REPRODUCIBILITY
    seed_everything(90)

    # CHECKPOINT HANDLING
    subfolder = args.model + '_' + args.version + "/"
    CHECKPOINT_PATH = root + "/trained_models/autoencoder/" + subfolder
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)

    # GET ESTIMATOR
    estim = get_estimator()  # returns blank estimator without initializations

    estim.yield_cell_type = True
    estim.yield_organ = True
    estim.yield_assay_sc = True
    estim.yield_data_batch = True
    estim.yield_tech_sample = True

    estim.init_datamodule(
        batch_size=args.batch_size,
        validation_batch_size=args.batch_size,
        retrieval_batch_size=args.RETRIEVAL_BATCH_SIZE,
        shuffle_buffer=args.SHUFFLE_BUFFER_SIZE,
        prefetch=ceil(args.RETRIEVAL_BATCH_SIZE / args.batch_size)
    )

    estim.init_trainer(
        trainer_kwargs={
            'max_epochs': 1000,
            'gradient_clip_val': 1.,
            'gradient_clip_algorithm': 'norm',
            'default_root_dir': CHECKPOINT_PATH,
            'resume_from_checkpoint': get_model_checkpoint(CHECKPOINT_PATH, args.resume_from_checkpoint),
            'accelerator': 'gpu',
            'devices': 1,
            'num_sanity_val_steps': 0,
            'check_val_every_n_epoch': 1,
            'limit_train_batches': int(len(estim.datamodule.idx_train) / args.batch_size),
            'limit_val_batches': int(len(estim.datamodule.idx_val) / args.batch_size),
            'logger': [TensorBoardLogger(CHECKPOINT_PATH, name='default')],
            'log_every_n_steps': 100,
            'detect_anomaly': False,
            'enable_progress_bar': True,
            'enable_model_summary': False,
            'enable_checkpointing': True,
            'callbacks': [
                TQDMProgressBar(refresh_rate=300),
                LearningRateMonitor(logging_interval='step'),
                # Save the model with the best training loss
                ModelCheckpoint(filename='best_checkpoint_train', monitor='train_loss_epoch', mode='min',
                                every_n_epochs=args.checkpoint_interval, save_top_k=1),
                # Save the model with the best validation loss
                ModelCheckpoint(filename='best_checkpoint_val', monitor='val_loss', mode='min',
                                every_n_epochs=args.checkpoint_interval, save_top_k=1)
            ],
        }
    )
    estim.init_model(
        model=args.model,
        model_kwargs={
            'units_encoder': hparams['units'],
            'units_decoder': hparams['units'][::-1][1:],
        }
    )
    print(ModelSummary(estim.model))

    # Run learning rate finder
    lr_find_kwargs = {'early_stop_threshold': None, 'min_lr': 1e-8, 'max_lr': 10., 'num_training': 120}

    lr_finder = estim.trainer.tuner.lr_find(estim.model, estim.datamodule, **lr_find_kwargs)

    # Results can be found in
    lr_finder.results

    # Plot with
    fig = lr_finder.plot(suggest=True)

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print('New lr: ', new_lr)

    # update hparams of the model
    estim.model.hparams.lr = new_lr

    estim.train()
