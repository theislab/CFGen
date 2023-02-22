celldreamer
=======

Generative modeling of single cell data.

Usage
-----
See the examplary notebook in the `notebooks` folder.

Installation
------------

1. Create the conda environment:

   .. code-block:: bash

       conda env create -f environment.yml

2. Activate the environment:

   .. code-block:: bash

       conda activate celldreamer

3. Install the package in development mode:

   .. code-block:: bash

       pip install -e .

4. Change the line in `/envs/celldreamer/lib/python3.9/site-packages/merlin/core/compat.py`
from

   .. code-block:: python

       HAS_GPU = nvml.device_get_count() > 0

to

   .. code-block:: python

       HAS_GPU = True

5. In the `.bashrc` and `.profile` files add the lines

   .. code-block:: bash
        
       export PATH=/usr/local/cuda-11/bin${PATH:+:${PATH}}
       export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
       export CUDA_HOME=/usr/local/cuda-11


Requirements
^^^^^^^^^^^^
See `environment.yml` for the required packages.

Compatibility
-------------
`celldreamer`is compatible with Python 3.9, not yet tested with Python 3.8.

Licence
-------
`celldreamer` is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.

Authors
-------

`celldreamer` was written by `Till Richter <till.richter@helmholtz-muenchen.de>`_, `Alessandro Palma  <alessandro.palma@helmholtz-muenchen.de>`_ and `Karsten Roth  <karsten.rh1@gmail.com>`_.
