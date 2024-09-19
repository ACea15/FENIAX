# Finite Element models 4 Intrinsic Nonlinear Aeroelastics in JAX

FENIAX is an aeroelastic toolbox  written in Python using JAX, which acts as a post-processor of commercial software such as MSC Nastran. 
Arbitrary FE models built for linear aeroelastic analysis are enhanced with geometric nonlinear effects, flight dynamics and linearized state-space solutions about nonlinear equilibrium.

!!! warning 
	The software is beta, and while it it is likely   not all Get in touch if you encounter problems



## Getting started

## Examples
The most relevant examples in the code base are shown here, these and more can be found in the folder `/examples`
They are also part of a large test suite that is integrated into the development using CI/CD.

!!! tip
    Navigate to the code of the various examples, including the simulation input settings and postprocessing of the simulation --exactly as it was used for the articles backing the software.


### Nonlinear structural static results
!!! success
    Validated with MSC Nastran nonlinear solution (sol 400)

[Notebook](./examples/SailPlane/sailplane_nb.md)

### Wing free dynamics
!!! success
    Validated with MSC Nastran nonlinear solution (sol 400)
	
[Notebook](./examples/wingSP/wingSP_nb.md)


![Wing free dynamics](./media/wingSP_optimized.gif)

	
### Free flying structure

[Bio](https://mechanics.stanford.edu/simo)

[Notebook](./examples/wingSP/wingSP_nb.md)

#### 2D dynamics
![Free flying structure 2D](./media/SimoFFB2D_optimized.gif)
#### 3D dynamics
![Free flying structure 3D](./media/SimoFFB3D_optimized.gif)

### Industrial Aircraft model
!!! success
    Linear response validated with MSC Nastran linear aeroelastic solution (sol 146)

#### Gust clamped model

[Notebook](./examples/XRF1/xrf1_nb.md)


![XRF1-gustclamped](./media/xrf1_gust_optimized.gif)


#### Gust trimmed flight
![XRF1-Trim+gust](./media/xrf1_trimgust_optimized.gif)

## Theoretical background

## Code base
	
## License
Please see the [project license](./LICENSE.md) for further details.
