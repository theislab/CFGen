from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = ROOT / "datasets"
RAW_DATA_DIR = Path("/lustre/groups/ml01/projects/2024_scvdm_till.richter/datasets/raw")
PROCESSED_DATA_DIR = Path("/lustre/groups/ml01/projects/2024_scvdm_till.richter/datasets/processed")
TRAINING_FOLDER = ROOT / "project_folder"
