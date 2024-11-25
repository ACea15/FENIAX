# Finite Element models 4 Nonlinear Intrinsic Aeroelastics in JAX [FENIAX]

FENIAX is an aeroelastic toolbox  written and parallelized in Python, which acts as a post-processor of commercial software such as MSC Nastran. 
Arbitrary FE models built for linear aeroelastic analysis are enhanced with geometric nonlinear effects, flight dynamics and linearized state-space solutions about nonlinear equilibrium.
Some of the key features of the software are:
- Leveraging on the numerical library JAX and optimised algorithms, a high performance is achieved that leads to simulation times comparable to the linear counterparts on conventional platforms.
- The software runs on modern hardware architectures such as GPUs in a addition to standard CPUs.
- Algorithm differentiation (AD) of the aeroelastic response is available via JAX primitives. 
- Concurrent simulations for multiple load cases are being developed.

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


## Documentation
Available at https://acea15.github.io/FENIAX/

## Simulation Examples

The most relevant examples in the code base are shown here, these and more can be found in the folder `/examples`
They are also part of a large test suite that is integrated into the development using CI/CD.

!!! tip
    Navigate to the code of the various examples, including the simulation input settings and postprocessing of the simulation --exactly as it was used for the articles backing the software.


### Nonlinear structural static results
!!! success
    Validated with MSC Nastran nonlinear solution (sol 400)

[Notebook](./docs/documentation/examples/SailPlane/sailplane_nb.md)

### Wing free dynamics
!!! success
    Validated with MSC Nastran nonlinear solution (sol 400)
	
[Notebook](./docs/documentation/examples/wingSP/wingSP_nb.md)


![Wing free dynamics](./docs/media/wingSP_optimized.gif)

	
### Free flying structure

[Bio](https://mechanics.stanford.edu/simo)

[Notebook](./docs/documentation/examples/wingSP/wingSP_nb.md)

#### 2D dynamics
![Free flying structure 2D](./docs/media/SimoFFB2D_optimized.gif)
#### 3D dynamics
![Free flying structure 3D](./docs/media/SimoFFB3D_optimized.gif)

### Industrial Aircraft model
!!! success
    Linear response validated with MSC Nastran linear aeroelastic solution (sol 146)

#### Gust clamped model

[Notebook](./docs/documentation/examples/XRF1/xrf1_nb.md)


![XRF1-gustclamped](./docs/media/xrf1_gust_optimized.gif)


#### Gust trimmed flight
![XRF1-Trim+gust](./docs/media/xrf1_trimgust_optimized.gif)

