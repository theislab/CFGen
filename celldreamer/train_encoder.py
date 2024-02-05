import hydra
import sys
from omegaconf import DictConfig
from celldreamer.estimator.encoder_estimator import EncoderEstimator

@hydra.main(config_path="../configs/configs_encoder", config_name="train", version_base=None)
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
    # Initialize estimator 
    estimator = EncoderEstimator(cfg)
    # Train the encoder (checkpoints automatically dumped)
    estimator.train()
    
if __name__ == "__main__":
    import traceback
    try:
        train()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
