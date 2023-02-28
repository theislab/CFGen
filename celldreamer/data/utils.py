# This code is based on sfairazero_benchmarks, but without OOD

import numpy as np
from os.path import join
import os

from os.path import join
from rdkit import Chem
import scanpy as sc
from typing import List, Optional, Union

indx = lambda a, i: a[i] if a is not None else None


class Args(dict):
    """
    Wrapper around a dictiornary to make its keys callable as attributes
    """
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        self.__dict__ = self


def canonicalize_smiles(smiles: Optional[str]):
    """
    Canonicalize the SMILES
    """
    if smiles:
        return Chem.CanonSmiles(smiles)
    else:
        return None

def drug_names_to_once_canon_smiles(
    drug_names: List[str], dataset: sc.AnnData, perturbation_key: str, smiles_key: str
    ):
    """
    Converts a list of drug names to a list of SMILES. The ordering is of the list is preserved
    """
    name_to_smiles_map = {
        drug: canonicalize_smiles(smiles)
        for drug, smiles in dataset.obs.groupby(
            [perturbation_key, smiles_key]
        ).groups.keys()
    }
    return [name_to_smiles_map[name] for name in drug_names]

def get_train_val_test_split(data_path: str = '/lustre/groups/ml01/workspace/felix.fischer.2/sfaira/data/store/'
                                              'dao_512_cxg_primary_subset_norm',
                             train_subsample: float = 1.,
                             val_subsample: float = 1.,
                             test_subsample: float = 1.):
    assert 0. <= train_subsample <= 1.
    assert 0. <= val_subsample <= 1.
    assert 0. <= test_subsample <= 1.
    # idxs a grouped by zarr chunks -> need to be sorted after subsetting
    train_idxs = np.load(os.path.join(data_path, 'train_val_test_splits', 'train.npy'))
    val_idxs = np.load(os.path.join(data_path, 'train_val_test_splits', 'val.npy'))
    test_idxs = np.load(os.path.join(data_path, 'train_val_test_splits', 'test.npy'))

    # don't use random sub-setting to keep zarr chunks
    # idxs are saved shuffled by chunk
    if train_subsample < 1.:
        train_idxs = train_idxs[:int(len(train_idxs) * train_subsample)]
    if val_subsample < 1.:
        val_idxs = val_idxs[:int(len(val_idxs) * val_subsample)]
    if test_subsample < 1.:
        test_idxs = test_idxs[:int(len(test_idxs) * test_subsample)]

    return {'train': np.sort(train_idxs), 'val': np.sort(val_idxs), 'test': np.sort(test_idxs)}


def get_store(path_train_store):
    train_store = (
        sfaira.data.load_store(
            cache_path=join(path_train_store, 'dao'),
            store_format='dao',
            columns=['assay_sc', 'organ', 'cell_type', 'cell_type_ontology_term_id', 'tech_sample']
        ).stores['Homo sapiens']
    )
    return train_store


def get_estimator(
        path_train_store='/lustre/groups/ml01/workspace/felix.fischer.2/sfaira/data/store/'
                         'dao_512_cxg_primary_subset_norm'
):

    train_store = get_store(path_train_store)

    # Estimator for training data
    estim = EstimatorAE(
        train_store,
        feature_means=np.load(join(path_train_store, 'normalization_data/zero_centering/means.npy')),
        train_val_test_splits=get_train_val_test_split(path_train_store)
    )
    return estim


def get_model_checkpoint(checkpoint_path, checkpoint):
    if checkpoint is None:
        return None
    else:
        return os.path.join(checkpoint_path, 'default', checkpoint)
