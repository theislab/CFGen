import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import scanpy as sc
import numpy as np
import os

def str_obs_to_int(adata, obs_key):
    """
    Convert string observations to integer classes
    """
    obs = adata.obs[obs_key]
    obs_unique = obs.unique()
    obs_unique.sort()
    obs_dict = dict(zip(obs_unique, range(len(obs_unique))))
    obs_int = obs.map(obs_dict)
    return obs_int


class HLCADataset(Dataset):
    """
    Create a PyTorch dataset given the train_adata.h5ad, val_adata.h5ad, and test_adata.h5ad in this directory
    The maximum size is the train_adata.h5ad with <9GB and should fit into memory.
    __getitem__ should return a batch of
    batch = {
            "X": batch of transcriptoms in adata.X,
            "y": {
            "y_BMI": corresponding BMI observations in adata.obs["BMI"] - 64 classes,
            "y_age_or_mean_of_age_range": corresponding age observation in adata.obs["age_or_mean_of_age_range"] - 52 classes,
            "y_ann_level_1": corresponding ann_level_1 observation in adata.obs["ann_level_1"] - 4 classes,
            # "y_ann_level_2": corresponding ann_level_2 observation in adata.obs["ann_level_2"] - 11 classes,
            # "y_ann_level_3": corresponding ann_level_3 observation in adata.obs["ann_level_3"] - 25 classes,
            # "y_ann_level_4": corresponding ann_level_4 observation in adata.obs["ann_level_4"] - 39 classes,
            # "y_ann_level_5": corresponding ann_level_5 observation in adata.obs["ann_level_5"] - 17 classes,
            "y_dataset": corresponding dataset observation in adata.obs["dataset"] - 14 classes,
            "y_lung_condition": corresponding lung_condition observation in adata.obs["lung_condition"] - 2 classes (['Healthy', 'Healthy (tumor adjacent)'])
            "y_smoking_status": corresponding smoking_status observation in adata.obs["smoking_status"] - 3 classes (['active', 'former', 'never'])
            "y_sex": corresponding sex observation in adata.obs["sex"] - 2 classes (['female', 'male'])
            "y_tissue": corresponding tissue observation in adata.obs["tissue"] - 3 classes (['nose', 'respiratory airway', 'lung parenchyma'])
            "y_self_reported_ethnicity": corresponding self reported ethnicity observation in adata.obs["self_reported_ethnicity“] - 7 classes (['European', 'Asian', 'African', 'admixed ancestry', 'American', 'Pacific Islander', 'unknown'])
            "y_development_stage": corresponding development stage observation in adata.obs["development_stage"] - 50 classes
            }
        }
    """
    def __init__(self, adata_path: str, mode: str = "train"):
        """
        HLCADataset constructor.

        Args:
            adata_path (str): AnnData object containing the data.
            mode (str): Mode of the dataset. Can be "train", "val", or "test".
            transform (callable): Optional transform to be applied on a sample.
        """
        self.mode = mode
        if self.mode == 'train':
            self.adata = sc.read_h5ad(os.path.join(adata_path, "train_adata.h5ad"))
        elif self.mode == 'val':
            self.adata = sc.read_h5ad(os.path.join(adata_path, "val_adata.h5ad"))
        elif self.mode == 'test':
            self.adata = sc.read_h5ad(os.path.join(adata_path, "test_adata.h5ad"))
        else:
            raise ValueError("Invalid mode. Must be train, val, or test.")

        # Convert categorical columns to integer classes
        categorical_columns = [
            # "BMI",
            # "age_or_mean_of_age_range",
            "ann_level_1",
            "dataset",
            "lung_condition",
            "smoking_status",
            "sex",
            "tissue",
            "self_reported_ethnicity",
            "development_stage"
        ]
        for column in categorical_columns:
            categories = self.adata.obs[column].astype(str).unique()
            mapping = {category: i for i, category in enumerate(categories)}
            self.adata.obs[column] = self.adata.obs[column].astype(str).map(mapping).astype(np.int64)


    def __len__(self):
        """
        Return the number of samples in the dataset.
        :return: int: Number of samples.
        """
        return self.adata.shape[0]

    def __getitem__(self, index):
        """
        Return a batch of samples from the dataset.
        :param index: Index of the sample.
        :return: dict: Batch of samples.
        """
        transcriptomes = self.adata.X[index]
        y_BMI = self.adata.obs["BMI"][index]
        y_age_or_mean_of_age_range = self.adata.obs["age_or_mean_of_age_range"][index]
        y_ann_level_1 = self.adata.obs["ann_level_1"][index]
        # y_ann_level_2 = self.adata.obs["ann_level_2"][index]
        # y_ann_level_3 = self.adata.obs["ann_level_3"][index]
        # y_ann_level_4 = self.adata.obs["ann_level_4"][index]
        # y_ann_level_5 = self.adata.obs["ann_level_5"][index]
        y_dataset = self.adata.obs["dataset"][index]
        y_lung_condition = self.adata.obs["lung_condition"][index]
        y_smoking_status = self.adata.obs["smoking_status"][index]
        y_sex = self.adata.obs["sex"][index]
        y_tissue = self.adata.obs["tissue"][index]
        y_self_reported_ethnicity = self.adata.obs["self_reported_ethnicity"][index]
        y_development_stage = self.adata.obs["development_stage"][index]

        batch = {
            "X": transcriptomes.toarray(),
            "y": {# "y_BMI": y_BMI,
                  # "y_age_or_mean_of_age_range": y_age_or_mean_of_age_range,
                  "y_ann_level_1": y_ann_level_1,
                  # "y_ann_level_2": y_ann_level_2,
                  # "y_ann_level_3": y_ann_level_3,
                  # "y_ann_level_4": y_ann_level_4,
                  # "y_ann_level_5": y_ann_level_5,
                  "y_dataset": y_dataset,
                  "y_lung_condition": y_lung_condition,
                  "y_smoking_status": y_smoking_status,
                  "y_sex": y_sex,
                  "y_tissue": y_tissue,
                  "y_self_reported_ethnicity": y_self_reported_ethnicity,
                  "y_development_stage": y_development_stage}
        }
        return batch