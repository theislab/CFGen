CFGen
=======

Generative modeling of single-cell data.

Installation
------------

1. Create the conda environment:

   .. code-block:: bash

       conda env create -f environment.yml

2. Activate the environment:

   .. code-block:: bash

       conda activate cfgen

3. Install the CFGen package in development mode:

   .. code-block:: bash

       cd directory_where_you_have_your_git_repos/cfgen
       pip install -e . 

4. Create symlink to the storage folder for experiments:

   .. code-block:: bash

       cd directory_where_you_have_your_git_repos/cfgen
       ln -s folder_for_experiment_storage project_folder


Requirements
^^^^^^^^^^^^
See `environment.yml` and `requirements.txt` for the required packages.

Compatibility
-------------
`cfgen` is compatible with Python 3.10.

Licence
-------
`cfgen` is licensed under the `MIT License <https://opensource.org/licenses/MIT>`_.
