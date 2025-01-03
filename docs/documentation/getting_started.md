# Getting Started

Welcome to the `Getting Started` guide for **FENIAX**.

## Installation

- Currently the code has been tested and is developed in Linux and MacOS.
- A minimum installation into the current environment is possible by navigating to the main directory and
```
pip install .
```

- However developer mode is recommended and also installing the full set of packages which include testing and visualisation capabilities:

```
pip install -e .[all]
```

- see pyproject.toml file for the options available. Python 3.10+ is required.

- To install with GPU support install jax first:
```
pip install -U "jax[cuda12]"
pip install -e ".[all]"
```


### Python environment
Although it is not necessary, it is recommended that the software is installed in its own environment. Options that have been tested out follow.

- _Conda_:

Although it is not necessary, If conda is being used as package manager, one can make a specific environment as,

```
conda create -n feniax python=3.11
conda activate feniax
```

If pytest has been installed, check everything is OK by running the tests: 

```
pytest
```

- Thus a typical installation would comprise of these 4 steps:
```
conda create -n feniax.python=3.11
conda activate feniax
pip install -e .[all]
pytest
```

- _pyenv_: Navigate to the root directory and run the following: 

```
  pyenv install 3.11.10
  pyenv virtualenv 3.11.10 feniax
  pyenv local feniax
  pip install -e .[all]
  pytest
```
By setting pyenv local to feniax, every time one moves to feniax directory the environment is automatically activated
