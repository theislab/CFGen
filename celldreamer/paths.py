import pathlib
from pathlib import Path

HOME_DIR = pathlib.Path.home()
PERT_DATA_DIR = HOME_DIR / "environment" / "celldreamer" / "datasets"
EMBEDDING_DIR = HOME_DIR / "environment" / "celldreamer" / "embeddings"