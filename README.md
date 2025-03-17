CFGen
=======

This is the official repository of the ICLR 2025 paper **"Multi-Modal and Multi-Attribute Generation of Single Cells with CFGen."** CFGen is a generative model for single-cell data that can generate single-modality RNA-seq data as well as multi-modal ATAC and RNA-seq data. CFGen preserves discreteness in the data by combining latent Flow Matching with discrete likelihood decoders for the individual modalities. Moreover, we introduce the concept of **multi-attribute generation**, where we steer our Flow Matching model to synthesize realistic cells according to multiple attributes (e.g., batch and cell type).  

We showcase two applications for the model:
* Data augmentation for enhanced cell type classification. 
* Batch correction. 

Find our paper at:

[ArXiv](https://arxiv.org/abs/2407.11734) 
[OpenReview](https://openreview.net/forum?id=3MnMGLctKb)

Installation
------------

1. Clone our repository 

```
git clone https://github.com/theislab/CFGen.git
```

2. Create the conda environment:

```
conda env create -f environment.yml
```

3. Activate the environment:

```
conda activate cfgen
```

3. Install the CFGen package in development mode:

```
cd directory_where_you_have_your_git_repos/cfgen
pip install -e . 
```

4. Create symlink to the storage folder for experiments:

```
cd directory_where_you_have_your_git_repos/cfgen
ln -s folder_for_experiment_storage project_folder
```

5. Create experiment and dataset folder. 

```
cd project_folder
mkdir datasets
mkdir experiments
```

Download the datasets from Zenodo and place them into datasets



Repository structure
------------
Set up structure of the `project_folder`.  

> Requirements

See `environment.yml` and `requirements.txt` for the required packages.


> Hydra

Our implementation leverages [hydra](https://hydra.cc/docs/intro/) to handle experiments. The configuration hierarchy can be found in the `configs` folder with two sub-folders:
* `configs_encoder`: configuration to train the autoencoder model with discrete decoder for each modality. 
* `configs_sccfm`: configuration to train the Flow Matching model leveraging a pre-trained autoencoder.  


> CFGen 

The training scripts are:
* `cfgen/train_autoencoder`: training script for the autoencoder. 
* `cfgen/train_sccfm`: script for training the Flow Matching model. 

> Models 

* The autoencoder model is implemented in `cfgen/models/base/encoder_model.py`. 
* The `cfgen/models/featurizer/` folder contains the interface to organize the featurization of the conditioning variables. 
* The Flow Matching model is implemented in `cfgen/models/fm/`. 

> Experiment class

The experiment classes for interacting with the encoder and Flow Matching model are in `cfgen/models/estimator/encoder_estimator.py` and `cfgen/models/estimator/cfgen_estimator.py`.

Training
------------
Training scripts are in `scripts` with sub-folders `train_autoencoders` and `train_cfgen`. To retrain the models, first train the autoencoder and then the CFGen model. The scripts by default assume the use of the `slurm` scheduling system. The scripts can be adapted to standard bash commands. 

### Autoencoder 
To train the autoencoder, two scripts are available:

* `train_slurm_autoencoder.sbatch`: scRNA-seq-only dataset. 
* `train_slurm_autoencoder_multimodal.sbatch`: PBMC10k multi-modal learning. 

To run the autoencoder training on a dataset, uncomment the lines associated to the dataset, and run the following command:

```
sbatch train_slurm_autoencoder.sbatch
```

or 

```
sbatch train_slurm_autoencoder_multimodal.sbatch
```

### CFGen 

Similar to the autoencoder model, train the Flow Matching model by running


```
sbatch train_fm_slurm_autoencoder.sbatch
```

or 

```
sbatch train_fm_slurm_autoencoder_multimodal.sbatch
```

Training with new dataset
------------

To train with a new data, pre-process the dataset and save it to a storage folder. Then, create a new configuration file in:

* `configs/configs_encoder/dataset/your_dataset_of_name`
* `configs/configs_sccfm/dataset/your_dataset_of_name`

Pointing at the dataset's path. Change the hyperparameters in the other hyperparameter files to accommodate the desired setting.

Then, add the following lines to the `config/scripts/train_autoencoder.sbatch` or `config/scripts/train_autoencoder_multimodal.sbatch`:

scRNA-seq:
```
python ../../cfgen/train_encoder.py dataset=your_dataset_name 
logger.project=your_dataset_name_project 
```

Multi-modal:
```
python ../../cfgen/train_encoder.py dataset=your_dataset_name 
encoder=encoder_multimodal logger.project=your_dataset_name_project 
```


Installation
------------
`cfgen` is compatible with Python 3.10.

```
@inproceedings{
    palma2025multimodal,
    title={Multi-Modal and Multi-Attribute Generation of Single Cells with {CFG}en},
    author={Alessandro Palma and Till Richter and Hanyi Zhang and Manuel Lubetzki and Alexander Tong and Andrea Dittadi and Fabian J Theis},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=3MnMGLctKb}
}
```

Compatibility
-------------
`cfgen` is compatible with Python 3.10.

Licence
-------
`cfgen` is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.