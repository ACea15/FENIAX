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

- see setup.py file for the options available. Python 3.9+ is required but 3.11+ is recommended. 
Although it is not necessary, If conda is being used as package manager, one can make a specific environment as,

```
conda create -n feniax python=3.11
conda activate fem4inas
```

- If pytest has been installed, check everything is OK by running the tests: 

```
pytest tests
```

- Thus a typical installation would comprise of these 4 steps:
```
conda create -n feniax.python=3.11
conda activate fem4inas
pip install -e .[all]
pytest tests
```

