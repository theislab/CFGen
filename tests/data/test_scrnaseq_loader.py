import pytest
import torch
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
from unittest.mock import patch
from celldreamer.data.scrnaseq_loader import RNAseqLoader
from celldreamer.data.utils import normalize_expression

@pytest.fixture
def mock_adata():
    adata = sc.AnnData(X=csr_matrix(np.random.rand(100, 100)))  # Creating a sparse matrix
    adata.obs["condition"] = np.random.choice(["A", "B"], size=100)
    return adata

@pytest.fixture
def rnaseq_loader(mock_adata):
    data_path = "test_data.h5ad"
    layer_key = "counts"
    covariate_keys = ["condition"]
    subsample_frac = 0.5
    encoder_type = "proportions"
    target_max = 1
    target_min = -1

    with patch("scanpy.read", return_value=mock_adata):
        loader = RNAseqLoader(data_path, layer_key, covariate_keys, subsample_frac, encoder_type, target_max, target_min)
    return loader

def test_init(rnaseq_loader, mock_adata):
    assert rnaseq_loader.encoder_type == "proportions"
    assert torch.all(torch.eq(rnaseq_loader.X, torch.Tensor(mock_adata.X.toarray())))  # Converting sparse matrix to dense
    assert torch.all(torch.eq(rnaseq_loader.X_norm, normalize_expression(rnaseq_loader.X, rnaseq_loader.X.sum(1).unsqueeze(1), "proportions")))
    assert rnaseq_loader.scaler is not None
    assert rnaseq_loader.id2cov == {"condition": {"A": 0, "B": 1}}
    assert rnaseq_loader.log_size_factor_mu is not None
    assert rnaseq_loader.log_size_factor_sd is not None
    assert rnaseq_loader.max_size_factor is not None
    assert rnaseq_loader.min_size_factor is not None

def test_get_scaler(rnaseq_loader):
    """
    Test the get_scaler method of the rnaseq_loader object.

    This test checks if the scaler object returned by the get_scaler method is not None.

    Args:
        rnaseq_loader: An instance of the rnaseq_loader object.

    Returns:
        None
    """
    scaler = rnaseq_loader.get_scaler()
    assert scaler is not None

def test_getitem(rnaseq_loader):
    """
    Test the __getitem__ method of the rnaseq_loader object.

    Args:
        rnaseq_loader: An instance of the rnaseq_loader object.

    Returns:
        None
    """
    item = rnaseq_loader[0]
    assert isinstance(item, dict)
    assert "X" in item
    assert "X_norm" in item
    assert "y" in item

def test_len(rnaseq_loader):
    """
    Test the length of the rnaseq_loader object.

    Parameters:
    rnaseq_loader (object): The rnaseq_loader object to be tested.

    Returns:
    None
    """
    length = len(rnaseq_loader)
    assert length == rnaseq_loader.X.shape[0]
