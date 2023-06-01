import logging
import os 
from typing import List, Optional, Union

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData

from sklearn.preprocessing import OneHotEncoder
from celldreamer.data.utils import indx, drug_names_to_once_canon_smiles

class PertDataset:
    covariate_keys: Optional[List[str]] 
    drugs: torch.Tensor
    drugs_idx: torch.Tensor 
    max_num_perturbations: int 
    dosages: torch.Tensor
    drugs_names_unique_sorted: np.ndarray

    def __init__(
        self,
        data: str,
        perturbation_key=None,
        dose_key=None,
        covariate_keys=None,
        smiles_key=None,
        degs_key="rank_genes_groups_cov",
        pert_category="cov_drug_dose_name",
        split_key='split',
        use_drugs=False,
    ):
        """
        Perturbation dataset 
        
        Args:
            data (str): name of the dataset 
            perturbation_key (str, optional): column of adata.obs with the perturbation names. Defaults to None.
            dose_key (str, optional): column of adata.obs with the dose values. Defaults to None.
            covariate_keys (str, optional): column of adata.obs with the covariate values. Defaults to None.
            smiles_key (str, optional): column of adata.obs with the SMILES strings. Defaults to None.
            degs_key (str, optional): entry of adata.obsm with differentially expressed genes. Defaults to "rank_genes_groups_cov".
            pert_category (str, optional): . defaults to "cov_drug_dose_name".
            split_key (str, optional): type of dataset splitting to experiment with. Defaults to 'split'.
            use_drugs (bool, optional): whether to use drug. Defaults to False.
        """
        # Read AnnData 
        assert os.path.exists(data)
        logging.info(f"Starting to read in data: {data}\n...")
        data = sc.read(data)
        logging.info(f"Finished data loading.")
        
        # Set AnnData fields as attributes 
        self.genes = torch.Tensor(data.X.A)
        self.var_names = data.var_names
        self.perturbation_key = perturbation_key
        self.dose_key = dose_key
        if isinstance(covariate_keys, str):
            covariate_keys = [covariate_keys]
        self.covariate_keys = covariate_keys
        self.smiles_key = smiles_key
        self.use_drugs = use_drugs
    
            
        # Extract relevant information from anndata
        self.pert_categories = np.array(data.obs[pert_category].values)  # (n_obs, 1) - perturbation categories 
        self.de_genes = data.uns[degs_key]  # Differential expressed genes per condition (drug+dose)
        self.drugs_names = np.array(data.obs[perturbation_key].values)  # (n_obs, 1) - name of drug used for each of the cells 
        self.dose_names = np.array(data.obs[dose_key].values)  # (n_obs, 1) - dose used to pertrub each cell 

        # Get unique drugs
        drugs_names_unique = set()  
        for d in self.drugs_names:
            [drugs_names_unique.add(i) for i in d.split("+")] 
        self.drugs_names_unique_sorted = np.array(sorted(drugs_names_unique))

        # Assign ID to drug names 
        self._drugs_name_to_idx = {
            smiles: idx for idx, smiles in enumerate(self.drugs_names_unique_sorted)
        }
        self.drug_dict = {val:key for key,val in self._drugs_name_to_idx.items()}

        # Collect smiles per drug 
        self.canon_smiles_unique_sorted = drug_names_to_once_canon_smiles(
            list(self.drugs_names_unique_sorted), data, perturbation_key, smiles_key
        )
        
        # Derive the maximum number of perturbations per cells in the dataset 
        self.max_num_perturbations = max(
            len(name.split("+")) for name in self.drugs_names
        )

        assert (
            self.max_num_perturbations == 1
        ), "Index-based drug encoding only works with single perturbations"
        
        # Extract the drug id for each observation 
        drugs_idx = [self.drug_name_to_idx(drug) for drug in self.drugs_names]
        self.drugs_idx = torch.tensor(
            drugs_idx,
            dtype=torch.long,
        )
        # Extract floats for the dosages 
        dosages = [float(dosage) for dosage in self.dose_names]
        self.dosages = torch.tensor(
            dosages,
            dtype=torch.float32,
        )

        # Prepare covariate query 
        if not len(covariate_keys) == len(set(covariate_keys)):
            raise ValueError(f"Duplicate keys were given in: {covariate_keys}")
        
        self.covariate_names = {}  
        self.covariate_names_unique = {}
        self.covariates = []
        
        # Derive one-hot encodings and names for covariates and their sub-categories
        for cov in self.covariate_keys:
            self.covariate_names[cov] = np.array(data.obs[cov].values)
            self.covariate_names_unique[cov] = np.unique(self.covariate_names[cov])
            names = self.covariate_names[cov]
            
            encoder_cov = OneHotEncoder(sparse=False)
            encoder_cov.fit(names.reshape(-1, 1)) 

            self.covariates.append(
                torch.Tensor(encoder_cov.transform(names.reshape(-1, 1))).float().argmax(1)
            )

        # Indicate if cell is control 
        self.ctrl = data.obs["control"].values

        # Indicate control name in the perturbation column
        if perturbation_key is not None:
            self.ctrl_name = list(
                np.unique(data[data.obs["control"] == 1].obs[self.perturbation_key])
            )
        else:
            self.ctrl_name = None

        # Number of covariates 
        if self.covariate_names_unique is not None:
            self.num_covariates = [
                len(names) for names in self.covariate_names_unique.values()
            ]
        else:
            self.num_covariates = [0]

        # Number of genes and drugs 
        self.num_genes = self.genes.shape[1]
        self.num_drugs = (len(self.drugs_names_unique_sorted))

        self.indices = {
            "all": list(range(len(self.genes))),
            "control": np.where(data.obs["control"] == 1)[0].tolist(),
            "treated": np.where(data.obs["control"] != 1)[0].tolist(),
            "train": np.where(data.obs[split_key] == "train")[0].tolist(),
            "test": np.where(data.obs[split_key] == "test")[0].tolist(),
            "ood": np.where(data.obs[split_key] == "ood")[0].tolist(),
        }

        # For each observation, get a bool vector to select in the gene dimension 
        degs_tensor = []
        for i in range(len(self)):
            drug = indx(self.drugs_names, i)
            cov = indx(self.covariate_names["cell_type"], i)

            if drug == "JQ1":
                drug = "(+)-JQ1"

            if drug == "control":
                genes = []
                
            else:
                genes = self.de_genes[f"{cov}_{drug}_1.0"]

            degs_tensor.append(
                torch.Tensor(self.var_names.isin(genes)).detach().clone()
            )
        self.degs = torch.stack(degs_tensor)

    def subset(self, split, condition="all"):
        """
        Subset the Dataset class given a pre-defined split 
        """
        idx = list(set(self.indices[split]) & set(self.indices[condition]))
        return SubPertDataset(self, idx)

    def drug_name_to_idx(self, drug_name: str):
        """
        For the given drug, return it's index. The index will be persistent for each dataset (since the list is sorted).
        Raises ValueError if the drug doesn't exist in the dataset.
        """
        return self._drugs_name_to_idx[drug_name]

    def __getitem__(self, i):
        X = self.genes[i]
        X_degs = indx(self.degs, i)
        if self.use_drugs:
            y = {"y_drug": [indx(self.drugs_idx, i), indx(self.dosages, i)]}
        else:
            y = {}
        y.update({"y_"+self.covariate_keys[cov_idx]: indx(cov, i) for cov_idx, cov in enumerate(self.covariates)})
        return ({"X": X, "X_degs": X_degs, "y": y})

    def __len__(self):
        return len(self.genes)


class SubPertDataset:
    """
    Subsets a `Dataset` by selecting the examples given by `indices`.
    """

    def __init__(self, dataset: PertDataset, indices):
        self.perturbation_key = dataset.perturbation_key
        self.dose_key = dataset.dose_key
        self.covariate_keys = dataset.covariate_keys
        self.smiles_key = dataset.smiles_key
        self.use_drugs = dataset.use_drugs

        self.genes = dataset.genes[indices]
        self.drugs_idx = indx(dataset.drugs_idx, indices)
        self.dosages = indx(dataset.dosages, indices)
        self.drugs_names = indx(dataset.drugs_names, indices)
        self.pert_categories = indx(dataset.pert_categories, indices)
        
        self.covariates = [indx(cov, indices) for cov in dataset.covariates]
        self.covariate_names = {}
        for cov in self.covariate_keys:
            self.covariate_names[cov] = indx(dataset.covariate_names[cov], indices)

        self.var_names = dataset.var_names
        self.de_genes = dataset.de_genes
        self.ctrl_name = indx(dataset.ctrl_name, 0)

        self.num_covariates = dataset.num_covariates
        self.num_genes = dataset.num_genes
        self.num_drugs = dataset.num_drugs

        self.degs = dataset.degs

    def __getitem__(self, i):
        X = self.genes[i]
        X_degs = indx(self.degs, i)
        if self.use_drugs:
            y = {"y_drug": [indx(self.drugs_idx, i), indx(self.dosages, i)]}
        else:
            y = {}
        y.update({"y_"+self.covariate_keys[cov_idx]: indx(cov, i) for cov_idx, cov in enumerate(self.covariates)})
        return ({"X": X, "X_degs": X_degs, "y": y})

    def __len__(self):
        return len(self.genes)

# Load the data splits
def load_dataset_splits(
    dataset_path: str,
    perturbation_key: Union[str, None],
    dose_key: Union[str, None],
    covariate_keys: Union[list, str, None],
    smiles_key: Union[str, None],
    degs_key: str = "rank_genes_groups_cov",
    pert_category: str = "cov_drug_dose_name",
    split_key: str = "split",
    return_dataset: bool = False,
    use_drugs=False,
):
    """Calls the dataset class and subsets it into splits 
    """
    dataset = PertDataset(
        dataset_path,
        perturbation_key,
        dose_key,
        covariate_keys,
        smiles_key,
        degs_key,
        pert_category,
        split_key,
        use_drugs,
    )

    splits = {
        "training": dataset.subset("train", "all"),
        "training_control": dataset.subset("train", "control"),
        "training_treated": dataset.subset("train", "treated"),
        "test": dataset.subset("test", "all"),
        "ood": dataset.subset("ood", "all"),
    }

    if return_dataset:
        return splits, dataset
    else:
        del dataset 
        return splits

if __name__ == '__main__':
    from celldreamer.paths import PERT_DATA_DIR
    from pathlib import Path
    path = Path(PERT_DATA_DIR)
    load_dataset_splits(
        dataset_path = path / 'sciplex' / 'sciplex_complete_middle_subset.h5ad',
        perturbation_key = 'condition',
        dose_key = 'dose',
        covariate_keys = 'cell_type',
        smiles_key = 'SMILES',
        degs_key = "lincs_DEGs",
        pert_category=  "cov_drug_dose_name",
        split_key = "split_ho_pathway",
        return_dataset = True)
    