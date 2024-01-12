import random

import pytest
from omegaconf import OmegaConf

import numpy as np
import pandas as pd

from anndata import AnnData


@pytest.fixture()
def adata():
    rng = np.random.default_rng()
    X_0 = rng.random((50, 10))
    X_1 = X_0 + 1

    X_pca = rng.random((100, 2))
    X = np.concatenate([X_0, X_1], axis=0)
    # Generate a list of 100 randomly chosen samples
    random_samples = [random.choice(["sample1", "sample2"]) for _ in range(100)]

    obs = {"time": np.concatenate([np.zeros(50), np.ones(50)]), "sample": random_samples}
    vars_df = pd.DataFrame(index=range(X.shape[1]))

    return AnnData(X=X, obs=pd.DataFrame(obs), obsm={"X_pca": X_pca}, var=vars_df)


@pytest.fixture()
def preprocessing_config():
    return OmegaConf.create(
        {
        # Placeholder for preprocessing config
        }
    )