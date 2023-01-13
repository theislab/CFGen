scgm
===========================

|PyPI| |Python Version| |License| |Read the Docs| |Build| |Tests| |Codecov| |pre-commit| |Black|

.. |PyPI| image:: https://img.shields.io/pypi/v/scgm.svg
   :target: https://pypi.org/project/scgm/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/scgm
   :target: https://pypi.org/project/scgm
   :alt: Python Version
.. |License| image:: https://img.shields.io/github/license/richtertill/scgm
   :target: https://opensource.org/licenses/MIT
   :alt: License
.. |Read the Docs| image:: https://img.shields.io/readthedocs/scgm/latest.svg?label=Read%20the%20Docs
   :target: https://scgm.readthedocs.io/
   :alt: Read the documentation at https://scgm.readthedocs.io/
.. |Build| image:: https://github.com/richtertill/scgm/workflows/Build%20scgm%20Package/badge.svg
   :target: https://github.com/richtertill/scgm/actions?workflow=Package
   :alt: Build Package Status
.. |Tests| image:: https://github.com/richtertill/scgm/workflows/Run%20scgm%20Tests/badge.svg
   :target: https://github.com/richtertill/scgm/actions?workflow=Tests
   :alt: Run Tests Status
.. |Codecov| image:: https://codecov.io/gh/richtertill/scgm/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/richtertill/scgm
   :alt: Codecov
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Black


Features
--------

* TODO

Setup
-----

Setup of the repository and conda environment
``git clone https://github.com/theislab/scgm.git``
``cd scgm``
``conda env create -f scgm.yml``
``source activate fa_base``
Setup of the ipykernel
``python -m ipykernel install --user --name scgm --display-name "scgm"``
Additional repositories to install
Sfaira
``cd target_directory``
``git clone https://github.com/theislab/sfaira.git``
``cd sfaira``
``git checkout b9f5e6b04e42bd6e650b353140829f0f5ae43060``
``git pull``
``pip install -e .``
Sfairazero
``cd target_directory``
``git clone https://github.com/theislab/sfairazero.git``
``cd sfairazero``
``git pull``
``pip install -e .``


Installation
------------

You can install *scgm* via pip_ from PyPI_:

.. code:: console

   $ pip install scgm


Usage
-----

Please see the `Command-line Reference <Usage_>`_ for details.


Credits
-------

This package was created with cookietemple_ using Cookiecutter_ based on Hypermodern_Python_Cookiecutter_.

.. _cookietemple: https://cookietemple.com
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _PyPI: https://pypi.org/
.. _Hypermodern_Python_Cookiecutter: https://github.com/cjolowicz/cookiecutter-hypermodern-python
.. _pip: https://pip.pypa.io/
.. _Usage: https://scgm.readthedocs.io/en/latest/usage.html
