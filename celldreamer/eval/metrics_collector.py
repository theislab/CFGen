import torch 
import numpy as np
from celldreamer.eval.evaluation_metrics.density_and_coverage import compute_prdc
from celldreamer.eval.evaluation_metrics.reconstruction_loss import reconstruction_loss
from celldreamer.eval.evaluation_metrics.evaluate_knn_graph import knn_graph_metric

class MetricsCollector:
    def __init__(self, 
                 dataloader,
                 task, 
                 feature_embeddings): 
        self.dataloader = dataloader
        self.task = task
        self.metric_dict = {}
        self.feature_embeddings = feature_embeddings
    
    def compute_generation_metrics(self, real_adata, reconstructed_adata, generated_adata):
        """
        Compute generation metrics 
        """
        cov_combinations = list(set(zip(*real_adata.obs.values.T)))

        colnames = real_adata.obs.columns
        for cov_combination in cov_combinations:
            selected_rows = real_adata.obs[colnames[0]] == cov_combination[0]
            for idx in range(len(cov_combination)-1):
                selected_rows = np.logical_and(selected_rows, real_adata.obs[colnames[idx]]== cov_combination[idx])
            
            tmp_metrics = {}
            tmp_metrics.update(reconstruction_loss(real_adata[selected_rows].X, reconstructed_adata[selected_rows].X))
            # tmp_metrics.update(knn_graph_metric(real_adata[selected_rows].X, generated_adata[selected_rows].X, k=5))
            tmp_metrics.update(compute_prdc(real_adata[selected_rows].X, generated_adata[selected_rows].X, nearest_k=5))
            self.update_metric_dict(tmp_metrics)
        
        # Take rthe average of the metrics 
        for key in self.metric_dict:
            self.metric_dict[key] = self.metric_dict[key]/len(real_adata)
    
    def update_metric_dict(self, tmp_metrics):
        if self.metric_dict == {}:
            self.metric_dict.update(tmp_metrics)
        else:
            for key in self.metric_dict:
                self.metric_dict[key] += tmp_metrics[key]
    
    def print_metrics(self):
        for key in self.metric_dict:
            print(f"Value for metric {key} is {self.metric_dict[key]}")
    
    def reset_metrics(self):
        self.metric_dict = {}
    