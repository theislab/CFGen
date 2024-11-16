import pandas as pd
import numpy as np
import scanpy as sc
import scvi
import torch
import scipy.special as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def main():
    # Read real datastet 
    dentate = sc.read_h5ad("/home/icb/alessandro.palma/environment/cfgen/project_folder/datasets/processed_full_genome/dentategyrus/dentategyrus_train.h5ad")
    dentate = corrupt_dataset(dentate)
    X_sparse = csr_matrix(dentate.layers["X_masked"])
    dentate.write_h5ad("/home/icb/alessandro.palma/environment/cfgen/project_folder/datasets/dropout/dentategyrus.h5ad")
    
    # Initialize scVI and load checkpoints 

    # Initialize CFGen 