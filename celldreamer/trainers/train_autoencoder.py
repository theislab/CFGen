import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from lightning_fabric.utilities.seed import seed_everything
from celldreamer.estimator.estimator import EstimatorAutoEncoder
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--model', type=str, default='mlp', help='model to use')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--units_encoder', type=list, default=[512, 512, 256, 256, 64],
                        help='list of units in encoder')
    parser.add_argument('--units_decoder', type=list, default=[64, 256, 256, 512, 512],
                        help='list of units in decoder')
    # saving parameter
    parser.add_argument('--version', type=str, default='', help='version of model')
    return parser.parse_args()


if __name__ == '__main__':
    # GET GPU AND ARGS
    if torch.cuda.is_available():
        print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    args = parse_args()

    # config parameters
    root = os.path.dirname(os.path.abspath(os.getcwd()))
    CHECKPOINT_PATH = os.path.join(root, 'trained_models/tb_logs', args.model)
    Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    LOGS_PATH = os.path.join(root, 'trained_models/tb_logs', args.model)
    DATA_PATH = '/lustre/scratch/users/felix.fischer/merlin_cxg_norm_parquet'

    # INIT ESTIMATOR
    estim = EstimatorAutoEncoder(DATA_PATH)
    seed_everything(1)

    # set data
    estim.yield_cell_type = True
    estim.yield_organ = True
    estim.yield_assay_sc = True
    estim.yield_data_batch = True
    estim.yield_tech_sample = True

    # set model
    estim.init_datamodule(batch_size=args.batch_size)

    estim.init_trainer(
        trainer_kwargs={
            'max_epochs': 1000,
            'gradient_clip_val': 1.,
            'gradient_clip_algorithm': 'norm',
            'default_root_dir': CHECKPOINT_PATH,
            'accelerator': 'gpu',
            'devices': 1,
            'num_sanity_val_steps': 0,
            'check_val_every_n_epoch': 1,
            'logger': [TensorBoardLogger(LOGS_PATH, name='default')],
            'log_every_n_steps': 100,
            'detect_anomaly': False,
            'enable_progress_bar': True,
            'enable_model_summary': False,
            'enable_checkpointing': True,
            'callbacks': [
                TQDMProgressBar(refresh_rate=100),
                LearningRateMonitor(logging_interval='step'),
                ModelCheckpoint(filename='best_train_loss', monitor='train_loss_epoch', mode='min',
                                every_n_epochs=1, save_top_k=1),
                ModelCheckpoint(filename='best_val_loss', monitor='val_loss', mode='min',
                                every_n_epochs=1, save_top_k=1)
            ],
        }
    )

    estim.init_model(
        model_type=args.model,
        model_kwargs={
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'lr_scheduler': torch.optim.lr_scheduler.StepLR,
            'units_encoder': args.units_encoder,
            'units_decoder': args.units_decoder,
            'lr_scheduler_kwargs': {
                'step_size': 2,
                'gamma': 0.9,
                'verbose': True
            },
            'optimizer': torch.optim.AdamW,
        },
    )

    print(ModelSummary(estim.model))
    # train model
    estim.train()
