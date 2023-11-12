import hydra
import sys
from omegaconf import DictConfig
from celldreamer.estimator.celldreamer_estimator import CellDreamerEstimator

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    """
    Main training function using Hydra.

    Args:
        cfg (DictConfig): Configuration parameters.

    Raises:
        Exception: Any exception during training.

    Returns:
        None
    """
    estimator = CellDreamerEstimator(cfg)
    estimator.train()
    estimator.test()

if __name__ == "__main__":
    import traceback
    try:
        train()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
