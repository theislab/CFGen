import numpy as np
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as skm
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd

def reconstruction_loss(X, X_reconstructed):
    mse = np.mean(np.mean((X-X_reconstructed)**2, axis=1))
    r2 = np.r2_score(X, X_reconstructed)
    return {"mse": mse, 
            "r2":r2}
