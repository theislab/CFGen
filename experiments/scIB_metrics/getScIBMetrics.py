#!/usr/bin/env python
import os
import sys
import argparse

import scanpy as sc
import scvi
import torch
from rich import print
from scib_metrics.benchmark import Benchmarker

# defaults
batch_key = "batch"      # batch key
n_hvg = 2000             # how many HVGs (comment if all genes shall be used)
layer_counts = "counts"  # count layer in adata objects
labels_key = "cell_type" # label to use for scANVI

SCVI_LATENT_KEY = "X_scVI"  # key of scvi latent representation
SCVI_MDE_KEY = "X_scVI_MDE" # key of mde embedding

SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_MDE_KEY = "X_scANVI_MDE"


def selectHVGs(adata):
    adata.raw = adata

    sc.pp.highly_variable_genes(
    adata,
    flavor="seurat_v3",
    n_top_genes=n_hvg,
    layer=layer_counts,
    batch_key=batch_key,
    subset=True
    )


def trainSCVI(adata):
    scvi.model.SCVI.setup_anndata(adata, layer=layer_counts, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    model.train()

    return model

def trainSCANVI(adata):
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key=labels_key,
        unlabeled_category="Unknown",
    )
    scanvi_model.train(max_epochs=20, n_samples_per_label=100)

    return scanvi_model

def plotLatentSCVI(adata):
    sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
    sc.tl.leiden(adata)
    adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata.obsm[SCVI_LATENT_KEY])

    sc.pl.embedding(
    adata,
    basis=SCVI_MDE_KEY,
    color=[batch_key, "leiden"],
    frameon=False,
    ncols=1,
    )

    sc.pl.embedding(adata, basis=SCVI_MDE_KEY, color=["cell_type"], frameon=False, ncols=1)

def plotLatentSCANVI(adata):
    adata.obsm[SCANVI_MDE_KEY] = scvi.model.utils.mde(adata.obsm[SCANVI_LATENT_KEY])
    sc.pl.embedding(
        adata, basis=SCANVI_MDE_KEY, color=["cell_type"], ncols=1, frameon=False
    )


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(
            prog="getScIBMetrics",
            description="Calculate the scIB metrics of scVI/scANVI models for a given dataset")

    parser.add_argument("adata")
    parser.add_argument("--checkpoint", help="path to load scVI/scANVI checkpoints")
    parser.add_argument("--save-checkpoint", help="path to save the scVI/scANVI checkpoint to")
    parser.add_argument("-s", "--save-results", help="save the results dataframe to file")
    parser.add_argument("-n", "--number-hvgs", type=int, default=n_hvg, help=f"how many HVGs to use. If set to 0, use all genes (default: {n_hvg})")
    parser.add_argument("-l", "--layer_counts", default=layer_counts, help=f"layer in the AnnData object that stores the raw counts (default: {layer_counts})")
    parser.add_argument("-c", "--labels_key", default=labels_key, help=f"labels to use for scANVI (default: {labels_key})")
    parser.add_argument("-k", "--batch_key", default=batch_key, help=f"batch key in AnnData object (default: {batch_key})")
    parser.add_argument("-g", "--graphical", action="store_true", default=False, help="display the results graphically")
    parser.add_argument("-p", "--plot-latents", action="store_true", default=False, help="plot some latent space visualizations")
    parser.add_argument("-o", "--overwrite", action="store_true", default=False, help="overwrite existing checkpoints")

    args = parser.parse_args()
    batch_key = args.batch_key
    n_hvg = args.number_hvgs
    layer_counts = args.layer_counts
    labels_key = args.labels_key

    if args.checkpoint and args.save_checkpoint:
        print("You can either load or save the checkpoints, not both")
        sys.exit()


    # Read data
    scvi.settings.seed = 0
    torch.set_float32_matmul_precision("high")

    adata = sc.read(args.adata)

    # Select HVGs
    if args.number_hvgs:
        selectHVGs(adata)

    # Train/Load scVI
    if args.checkpoint:
        model = scvi.model.SCVI.load(os.path.normpath(args.checkpoint_scvi) + "_scVI", adata)
    else:
        model = trainSCVI(adata)

    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation(adata)

    if args.save_checkpoint:
        model.save(os.path.normpath(args.save_checkpoint) + "_scVI", overwrite=args.overwrite)

    if args.plot_latents:
        plotLatentSCVI(adata)

    # Train/Load scANVI
    if args.checkpoint:
        scanvi_model = scvi.model.SCANVI.load(os.path.normpath(args.checkpoint) + "_scANVI", adata)
    else:
        scanvi_model = trainSCANVI(adata)
    
    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)

    if args.save_checkpoint:
        scanvi_model.save(os.path.normpath(args.save_checkpoint) + "_scANVI", overwrite=args.overwrite)

    if args.plot_latents:
        plotLatentSCANVI(adata)


    # Calculate benchmarks
    bm = Benchmarker(adata, batch_key=batch_key, label_key="cell_type", embedding_obsm_keys=["X_pca", SCVI_LATENT_KEY, SCANVI_LATENT_KEY], n_jobs=-1)
    bm.benchmark()

    if args.save_results:
        bm.get_results(min_max_scale=False).to_csv(args.save_results)

    if args.graphical:
        bm.plot_results_table(min_max_scale=False)
    else:
        print(bm.get_results(min_max_scale=False))
