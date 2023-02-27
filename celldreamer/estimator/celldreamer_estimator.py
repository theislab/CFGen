# Adapted from:
# https://github.com/theislab/cellnet/blob/main/cellnet/estimators.py
from os.path import join
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from cellnet.datamodules import MerlinDataModule
from cellnet.models import TabnetClassifier
from celldreamer.models.generative_models.autoencoder import MLP_AutoEncoder

class CellDreamerEstimator:
    