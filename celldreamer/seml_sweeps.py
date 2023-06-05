from sacred import Experiment
import seml
from celldreamer.data.utils import Args
from celldreamer.estimator.celldreamer_estimator import CellDreamerEstimator
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

class ExperimentWrapper:
    @ex.capture(prefix="config")
    def train(self, args):
        args = Args(args)  
        
        # Initialize solver with the defined arguments 
        estimator = CellDreamerEstimator(args)
        estimator.train()

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper()
    return experiment

# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    experiment.train()