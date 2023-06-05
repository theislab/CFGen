import wandb
from wandb.sweeps import GridSearch
import submitit

# Initialize wandb
wandb.init()

# Define hyperparameters to optimize
hyperparameter_defaults = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'num_layers': 2
}
wandb.config.update(hyperparameter_defaults)

