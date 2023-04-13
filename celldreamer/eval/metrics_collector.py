import torch 
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
    
    def compute_generation_metrics(self, real, reconstructed, generated):
        """
        Compute generation metrics 
        """
        for key in self.real:
            tmp_metrics = {}
            tmp_metrics.update(reconstruction_loss(real[key], reconstructed[key]))
            tmp_metrics.update(knn_graph_metric(real[key], generated[key]))
            tmp_metrics.update(compute_prdc(real[key], generated[key]))
            self.update_metric_dict(tmp_metrics)
        
        for key in self.metric_dict:
            self.metric_dict[key] = self.metric_dict[key]/len(self.real)
    
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
    