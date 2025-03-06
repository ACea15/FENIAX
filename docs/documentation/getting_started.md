# Getting Started

Welcome to the `Getting Started` guide for **FENIAX**.

## Installation

- Currently the code has been tested and is developed in Linux and MacOS. Get the code from GitHub first, and move to the directory:
```
git clone https://github.com/ACea15/FENIAX.git
cd FENIAX
```
- see pyproject.toml file for the options available. Python 3.10+ is required.
- A minimum installation into the current environment is possible by navigating to the main directory and
```
pip install .
```

However developer mode is recommended and also installing the full set of packages which include testing and visualisation capabilities:

```
pip install -e .[all]
```
or do the following if no visualisation is required or visualisation libraries give issues:
```
pip install -e .[dev]
```

- check everything is OK by running the tests: 

```
pytest
```


- To install with GPU support install jax first:
```
pip install -U "jax[cuda12]"
pip install -e ".[all]"
```

### Python environment
Although it is not necessary, it is recommended that the software is installed in its own environment. Options that have been tested out follow.

#### Venv
venv is the python native virtual environment. In the root of the project directory run 
```
python -m venv name_env
source name_env/bin/activate
```
Then install as above and all the required packages will be installed in the local bin folder. As always pytest to check everything runs fine.
#### UV
UV is a new python package manager, very fast as it is written in Rust. Once installed, the workflow is similar to that of Venv. 
Install the python version you want to use:
```
uv python install 3.11
```
Now similar to venv, create a folder in the project root, activate the environment, install the code (note uv comes first than pip and a python version is selected as many might be available), run tests:
```
uv venv name_env --python 3.11
source name_env/bin/activate
uv pip install -e .[all]
pytest
```

#### Pyenv
pyenv is a good way to maintain various python version in your system, lighter than conda and also provides a centralised solution for all your virtual environments.
Navigate to the root directory and run the following: 

```
  pyenv install 3.11.10
  pyenv virtualenv 3.11.10 feniax
  pyenv local feniax
  pip install -e .[all]
  pytest
```
By setting pyenv local to feniax, every time one moves to feniax directory the environment is automatically activated

#### Conda

If conda is being used as package manager, one can make a specific environment as,

```
conda create -n feniax python=3.11
conda activate feniax
```

Thus a typical installation would comprise of these 4 steps:
```
conda create -n feniax.python=3.11
conda activate feniax
pip install -e .[all]
pytest
```

