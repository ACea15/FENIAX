# Crossbeam test case with FENIAX

This notebook takes you through the steps of using FENIAX to simulation a flexible crossbeam. The test case is taken from Hilger (2024), [_A Dynamic Nonlinear Inertia Relief Formulation_](https://doi.org/10.2514/6.2024-2258). It involves a crossbeam under sinusoidal forcing vertically and has been validated with other nonlinear formulations. 

## Load modules
Here we load the relevant modules used in the simulation.

``` python
import feniax.preprocessor.configuration as configuration  
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pathlib
import numpy as np
```

## Housekeeping
We then define a few useful variables. 

``` python
gravity_forces = False
gravity_label = "g" if gravity_forces else ""
label = 'm1'
label_name = label + gravity_label
```

## Case setup
We now set up a FENIAX input object for an intrinsic modal dynamic simulation.

``` python
inp = Inputs()
# inp.log.level="debug"
inp.engine = "intrinsicmodal"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_{label_name}")
```

### Finite-element model file inputs
As part of the input we have a few choices in how to supply the the finite-element model files. 

Here the "inputs" flag is used to specify eigenvector and eigenvalue input by file paths. To produce these files refer to the useful functions `feniax.unastran.op2reader` and `feniax.unastran.matrixbuilder` for extracting the eigenvectors and eigenvalues from `.op2` files; and the mass and stiffness matrices from `.pch` files respectively. `num_modes` is used to limit number of modes used in the simulation.

Connectivity is defined a priori in the description of the loads paths `BuildAsetModel` from `feniax.unastran.asetbuilder` and is supplied via `grid`.

``` python
inp.fem.eig_type = "inputs"
inp.fem.Ka_name = f"./FEM/Ka_{label}.npy"
inp.fem.Ma_name = f"./FEM/Ma_{label}.npy"
inp.fem.eig_names = [f"./FEM/eigenvals_{label}.npy",
                     f"./FEM/eigenvecs_{label}.npy"]
inp.fem.num_modes = 60

inp.fem.connectivity = {'rbeam': None, 'lbeam': None, 'ubeam': None, 'dbeam': None}
inp.fem.grid = f"./FEM/structuralGrid_{label}"
```

### Boundary conditions
We then define the boundary conditions for the dynamic simulation. `bc1` can be "clamped" or "free" for the spatial boundary condition, whereas two of `t1`, `tn`, `dt` has to be set to define the temporal boundary condition. `t0` defaults to 0 if not specified. 

``` python
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.t1 = 1.0
inp.systems.sett.s1.dt = 0.0001
# inp.system.tn = 10000 + 1
```

### Solver settings
A number of different solvers are available. A Runge-Kutta routine is available via "runge_kutta"; for the [_Diffrax_](https://docs.kidger.site/diffrax) library use "diffrax".

``` python
inp.systems.sett.s1.solver_library = "diffrax" #
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")
```

### Load cases
To define the load conditions, use `gravity_forces` to toggle gravity. Here we opt for a dead force applied at the root node `0` in the vertical component `2`. The load profile is interpolated from the supplied array. 

``` python
inp.systems.sett.s1.xloads.gravity_forces = gravity_forces
inp.systems.sett.s1.xloads.dead_forces = True
inp.systems.sett.s1.xloads.dead_points = [[0, 2]]
t = np.linspace(0.0, 1.0, 10001)
freq = 2
amp = 0.1
f = amp * np.sin(2*np.pi*freq*t)
inp.systems.sett.s1.xloads.x = t.tolist()
inp.systems.sett.s1.xloads.dead_interpolation = [f.tolist()]
```

### Run FENIAX
And last but not least, to supply the `config` object and launch FENIAX.

``` python
config =  configuration.Config(inp)
sol = feniax.feniax_main.main(input_obj=config)
```

## Postprocessing
Here we wish to extract the position and velocity of all beam nodes into csv files; `feniax.preprocessor.solution.IntrinsicReader` is used to load the solution and `feniax.plotools.utils.pickIntrinsic2D` is used to extract the requested data and return arrays.

``` python
import os
import numpy as np
import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution

sol0 = solution.IntrinsicReader("./results_m1")

route_test_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
route_export = os.path.join(route_test_dir, "csvoutput")
os.makedirs(route_export, exist_ok=True)


def export_data(x_data, y_data, axis, route_export, file_prefix):
    n_nodes = y_data.shape[axis]

    for node_idx in range(n_nodes):
        x, y = putils.pickIntrinsic2D(
            x_data,
            y_data,
            fixaxis2=dict(node=node_idx, dim=axis)
        )

        dest_file = os.path.join(route_export, f"{file_prefix}_node_{node_idx}.txt")
        np.savetxt(
            dest_file,
            np.column_stack((x, y)),
            delimiter=","
        )

        print(f"Saved {file_prefix} node {node_idx} to {dest_file}")

    y_all = y_data[:, axis, :]
    out = np.column_stack((x_data, y_all))

    dest_file = os.path.join(route_export, f"{file_prefix}_all_nodes.txt")
    np.savetxt(dest_file, out, delimiter=",")

    print(f"Saved all nodes to {dest_file}")
```

### Export csv files
And hence the calls are made as such to extract first the velocities then the positions, in the z-direction, as below.

``` python
x_data = sol0.data.dynamicsystem_s1.t

export_data(
    x_data,
    sol0.data.dynamicsystem_s1.X1,
    2,
    route_export,
    "vel_z"
)

export_data(
    x_data,
    sol0.data.dynamicsystem_s1.ra,
    2,
    route_export,
    "pos_z"
)
```
