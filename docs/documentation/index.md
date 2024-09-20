# Finite Element models 4 Intrinsic Nonlinear Aeroelastics in JAX

FENIAX is an aeroelastic toolbox  written in Python using JAX. It acts as a post-processor of commercial software such as MSC Nastran. 

- Arbitrary FE models built for linear aeroelastic analysis are enhanced with geometric nonlinear effects, flight dynamics and linearized state-space solutions about nonlinear equilibrium.
- Nonlinear solutions run very fast, at or close to real time.
- Algorithmic differentiation (AD) of the response is available via JAX. The code is carefully crafted to perform all computations on tensor data structures and via algorithms available for AD, much like Machine Learning models are built.
- The code can be run on modern hardware architectures such as GPUs.

!!! warning 
	The software is in beta, and while it has been thoroughly tested, new features keep being added and it is likely features for your analysis might be missing. Get in touch if you encounter problems.



## Getting started
If you just want to start running the code, navigate to the [Getting started](./getting_started.md)
## Examples
The most relevant examples in the code base are shown here, these and more can be found in the folder `/examples`
They are also part of a large test suite that is integrated into the development using CI/CD.

!!! tip
    Navigate to the code of the various examples, including the simulation input settings and postprocessing of the simulation --exactly as it was used for the articles backing the software. See [examples](./examples.md)


### Nonlinear structural static results
!!! success
    - Validated with MSC Nastran nonlinear solution (sol 400)
	- AD differentiation of the response verified against finite-differences
	

[Notebook](./examples/SailPlane/sailplane_nb.md)

![Sail Plane static](./img/SailPlane3D_front.png)

!!! note
    Take a liner FE model of arbitrary complexity from your favourite FE solver, and turn it into a fully geometrically nonlinear model. You just need a condensation step into the main load paths and the resulting linear stiffness and mass matrices.  
### Wing free dynamics
!!! success
    - Validated with MSC Nastran nonlinear solution (sol 400)
    - Runs over x100 faster than Nastran 
    - AD differentiation of the response verified against finite-differences

[Notebook](./examples/wingSP/wingSP_nb.md)


![Wing free dynamics](./media/wingSP_optimized.gif)

	
### Free flying structure
This example first appeared in the work of Juan Carlos Simo (see [Bio](https://mechanics.stanford.edu/simo))
, a pioneer in the field of computational structural mechanics and the 

[Notebook](./examples/wingSP/wingSP_nb.md)

#### 2D dynamics
![Free flying structure 2D](./media/SimoFFB2D_optimized.gif)
#### 3D dynamics
![Free flying structure 3D](./media/SimoFFB3D_optimized.gif)

### Industrial Aircraft model
!!! success
    - Linear response validated with MSC Nastran linear aeroelastic solution (sol 146)
	- Nonlinear response in our solvers takes similar times to the linear Nastran solution!! 

#### Gust clamped model

[Notebook](./examples/XRF1/xrf1_nb.md)


![XRF1-gustclamped](./media/xrf1_gust_optimized.gif)


#### Gust trimmed flight
![XRF1-Trim+gust](./media/xrf1_trimgust_optimized.gif)

## Theoretical background

## Code base
	
## License
Please see the [project license](./LICENSE.md) for further details.
