from os.path import join
from typing import Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from cellnet.datamodules import MerlinDataModule