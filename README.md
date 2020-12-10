[![DOI](https://www.zenodo.org/badge/212395664.svg)](https://www.zenodo.org/badge/latestdoi/212395664)

## Code to accompany *[Efficient training of energy-based models via spin-glass control](https://www.arxiv.org/abs/1910.01592)*
#### Alejandro Pozas-Kerstjens, Gorka Muñoz-Gil, Miguel Angel García-March, Antonio Acín, Maciej Lewenstein, and Przemyslaw R. Grzybowski

This is a repository containing the code for models with Restricted Axons and training via Pattern-InDuced correlations (RAPID), developed in the article "*Efficient training of energy-based models via spin-glass control*. Alejandro Pozas-Kerstjens, Gorka Muñoz-Gil, Eloy Piñol, Miguel Ángel García-March, Antonio Acín, Maciej Lewenstein, and Przemysław R. Grzybowski. [arXiv:1910.01592](https://www.arxiv.org/abs/1910.01592)."

All code is written in Python.

Libraries required:
- [ebm-torch](https://github.com/apozas/ebm-torch) for energy-based models
- [gc](https://docs.python.org/3/library/gc.html) for garbage collection
- [itertools](https://docs.python.org/2/library/itertools.html) for combinatorial operations
- [math](https://docs.python.org/3/library/math.html) and [numpy](http://www.numpy.org/) for math operations
- [matplotlib](https://matplotlib.org/) for plot generation
- [pytorch](http://www.pytorch.org) >= 0.4.0 as ML framework
- [tqdm](https://pypi.python.org/pypi/tqdm) for custom progress bar

Files: 

  - [comparison](https://github.com/apozas/rapid/blob/master/comparison.py): example of use. Trains restricted Boltzmann machines using commonplace samplers and via RAPID, and computes various indicators of quality of training.
  
  - [MNIST training](https://github.com/apozas/rapid/blob/master/MNIST%20training.ipynb): example of use. Trains an RA-RBM in the MNIST dataset.

  - [rapid](https://github.com/apozas/rapid/blob/master/rapid.py): contains the relevant classes.
  
  - [utils](https://github.com/apozas/rapid/blob/master/utils.py): additional functions relevant for the example. 
