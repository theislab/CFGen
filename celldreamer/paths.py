import pathlib
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

PERT_DATA_DIR = ROOT / "datasets"
EMBEDDING_DIR = ROOT / "embeddings"
TRAINING_FOLDER = ROOT / "checkpoints"