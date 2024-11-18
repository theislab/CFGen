import torch
import scanpy as sc
from scib_metrics.benchmark import Benchmarker
import argparse 
from pathlib import Path
import os
from celldreamer.paths import DATA_DIR

def get_args():
    parser = argparse.ArgumentParser(description="RNA-seq Data Correction Pipeline")
    
    # Dataset and file paths
    parser.add_argument('--adata_path', type=str, required=True, 
                        help="Name of the dataset to process (subdirectory in 'processed_full_genome').")
    parser.add_argument('--generated_adata_path', type=str, required=True, 
                        help="Name of the dataset to process (subdirectory in 'processed_full_genome').")
    parser.add_argument('--batch_key', type=str, required=True, 
                        help="Name of the dataset to process (subdirectory in 'processed_full_genome').")
    parser.add_argument('--bio_key', type=str, required=True, 
                        help="Name of the dataset to process (subdirectory in 'processed_full_genome').")
    # Parse arguments
    args = parser.parse_args()    
    return args

def main(args):
    # Path to the anndata of origin 
    adata_original = sc.read_h5ad(DATA_DIR / 'processed_full_genome' / args.adata_path / f'{args.adata_path}.h5ad')
    # Path to the generated anndatas 
    generated_adata_path = Path(args.generated_adata_path)

    # Read the original dataset 
    embedding_obsm_keys = []
    for adata_path in os.listdir(generated_adata_path):
        adata_name = adata_path.split(".")[0]
        adata_corrected = sc.read_h5ad(generated_adata_path / adata_path)
        adata_original.obsm[adata_name] = adata_corrected.X.copy()
        embedding_obsm_keys.append(adata_name)
        
    # Benchmarker 
    bm = Benchmarker(adata_original,
                        batch_key=args.batch_key,
                        label_key=args.bio_key,
                        embedding_obsm_keys=embedding_obsm_keys,
                        n_jobs=-1)
    bm.benchmark()
    df = bm.get_results(min_max_scale=False)
    df = df.T.groupby("Metric Type").mean() 
    
    df.to_csv(f"/home/icb/alessandro.palma/environment/cfgen/notebooks/experiments/multi_label/{args.adata_path}.csv")

if __name__ == "__main__":
    args = get_args()
    main(args)
    