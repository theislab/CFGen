import hydra
import sys
from omegaconf import DictConfig
from celldreamer.estimator.celldreamer_estimator import CellDreamerEstimator

@hydra.main(config_path="../configs/configs_sccfm", config_name="train", version_base=None)
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
    estimator = CellDreamerEstimator(cfg)
    # Train and test 
    estimator.train()
    estimator.test()
    # Get test metric dictionary
    metrics_dict = estimator.trainer_generative.callback_metrics
    # Retrurn test metric (if any) for hparam tuning
    test_metric = cfg.get("optimized_metric")  # Assifgns None if not initialized 
    if test_metric:
        return metrics_dict[test_metric]
    else:
        return None
    
if __name__ == "__main__":
    import traceback
    try:
        train()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
