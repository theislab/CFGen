import os
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary

from celldreamer.paths import ROOT
from celldreamer.estimator.celldreamer_estimator import CellDreamerEstimator
from celldreamer.paths import DATA_DIR
from celldreamer.data.utils import Args


def train():
    """
    Train the diffusion model on HLCA data
    :return: estimator
    """
    # args
    config = yaml.safe_load(open(ROOT / "configs/hlca/config_ddpm.yaml",
                                 "rb"))
    args_hlca = Args(config["args"])

    # set up environment
    if torch.cuda.is_available():
        print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    estimator = CellDreamerEstimator(args_hlca)

    estimator.train()

    return estimator


def generate(estimator, sample_size):
    """
    Generate samples from the trained model
    :param estimator: estimator wrapper
    :param sample_size: number of samples to generate
    :return: X_gen, generated data
    """
    # args
    config = yaml.safe_load(open(ROOT / "configs/hlca/config_ddpm.yaml",
                                 "rb"))
    args_hlca = Args(config["args"])

    # generate samples
    X_gen= estimator.generative_model.ddim_sample(batch_size=sample_size,
                                                  y=None,
                                                  return_all_timesteps = False,
                                                  ddim_sampling_eta=0)
    # save generated samples
    experiment_name = args_hlca.experiment_name
    np.save(DATA_DIR / "generated_samples/hlca/X_gen_" + experiment_name + ".npy", X_gen)

    return X_gen


def plot_joint_adata(estimator, X_gen):
    """
    Plot a joint adata object with the original data and the generated data
    :param estimator: estimator wrapper
    :param X_gen: generated data
    :return:
    """
    # args
    config = yaml.safe_load(open(ROOT / "configs/hlca/config_ddpm.yaml",
                                    "rb"))
    args_hlca = Args(config["args"])
    experiment_name = args_hlca.experiment_name

    # get real data
    d = []

    for batch in estimator.datamodule.train_dataloader:
        d.append(batch["X"])

    d = torch.cat(d, dim=0)

    # get generated data
    adata = sc.AnnData(X = np.concatenate([X_gen.detach().cpu().numpy(), d.cpu().numpy()]),
                       obs = pd.DataFrame({"type":["gen"]*len(X_gen)+["real"]*len(d)}))

    # plot
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    # save umap

    # pca with type as color
    sc.pl.pca(adata, color="type", save="_pca_" + experiment_name + ".pdf")

    sc.pl.umap(adata, color="type", save="_umap_" + experiment_name + ".pdf")


def main():
    """
    Train the diffusion model on HLCA data and generate samples
    :return:
    """
    # train
    estimator = train()

    # generate
    X_gen = generate(estimator, sample_size=1000)

    # plot
    plot_joint_adata(estimator, X_gen)


if __name__ == "__main__":
    main()
