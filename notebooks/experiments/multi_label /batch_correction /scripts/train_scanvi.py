import numpy as np
import pandas as pd
import yaml
import torch
import scanpy as sc
import scvi
import scib 
from scarches.models.scpoli import scPoli

from scib_metrics.benchmark import Benchmarker

from copy import deepcopy
import matplotlib.pyplot as plt
from celldreamer.eval.eval_utils import normalize_and_compute_metrics

from celldreamer.paths import DATA_DIR

from celldreamer.models.fm.ode import torch_wrapper
from tqdm import tqdm 
from pathlib import Path

device  = "cuda" if torch.cuda.is_available() else "cpu"

adata = sc.read_h5ad(DATA_DIR / 'processed_full_genome' / 'c_elegans' / 'c_elegans.h5ad')

scvi.model.SCVI.setup_anndata(adata, layer="X_counts", batch_key="batch")

model = scvi.model.SCVI(adata, n_layers=2, n_latent=50, gene_likelihood="nb")

scanvi_model = scvi.model.SCANVI.from_scvi_model(
    model,
    adata=adata,
    labels_key="cell_type",
    unlabeled_category="Unknown"
)

scanvi_model.train(max_epochs=100, batch_size=256, early_stopping=True)

scanvi_model.save("/home/icb/alessandro.palma/environment/cfgen/project_folder/baseline_experiments/batch_correction/scanvi_model_celegans")