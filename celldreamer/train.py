import yaml 

from celldreamer.estimator.celldreamer_estimator import CellDreamerEstimator
from celldreamer.paths import PERT_DATA_DIR
from celldreamer.data.utils import Args

def train(args):
    estimator = CellDreamerEstimator(args)
    estimator.train()

if __name__=="__main__":
    import argparse
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse configuration file")
    
    # Add arguments
    parser.add_argument('-f', '--file', type=str, help='Path to yaml configuration')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read config file 
    with open(args.pat, "r") as stream:
        args = yaml.safe_load(args)
    
    estimator = CellDreamerEstimator(args)
    
    estimator.train()
     