
# Load modules

``` python
import plotly.express as px
import pyNastran.op4.op4 as op4
import matplotlib.pyplot as plt
import pdb
import datetime
import os
import shutil
REMOVE_RESULTS = True
#   for root, dirs, files in os.walk('/path/to/folder'):
#       for f in files:
#           os.unlink(os.path.join(root, f))
#       for d in dirs:
#           shutil.rmtree(os.path.join(root, d))
# 
if os.getcwd().split('/')[-1] != 'results':
    if not os.path.isdir("./figs"):
        os.mkdir("./figs")
    if REMOVE_RESULTS:
        if os.path.isdir("./results"):
            shutil.rmtree("./results")
    if not os.path.isdir("./results"):
        print("***** creating results folder ******")
        os.mkdir("./results")
    os.chdir("./results")
```

``` {#PYTHONMODULES .python}
import pathlib
import plotly.express as px
import pickle
import jax.numpy as jnp
import jax
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.uplotly as uplotly
import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader
import feniax.plotools.nastranvtk.bdfdef as bdfdef
from tabulate import tabulate
```

# Run models

``` python

import time

TIMES_DICT = dict()
SOL = dict()
CONFIG = dict()

def run(input1, **kwargs):
    jax.clear_caches()
    label = kwargs.get('label', 'default')
    t1 = time.time()
    config =  configuration.Config(input1)
    sol = feniax.feniax_main.main(input_obj=config)
    t2 = time.time()
    TIMES_DICT[label] = t2 - t1      
    SOL[label] = sol
    CONFIG[label] = config

def save_times():
    pd_times = pd.DataFrame(dict(times=TIMES_DICT.values()),
                            index=TIMES_DICT.keys())
    pd_times.to_csv("./run_times.csv")
```

-   Models run on this exercise:

      Label   Model           NumModes   Solver                        tol/dt       settings
      ------- --------------- ---------- ----------------------------- ------------ ----------------------
      SP1     SailPlane       5          Newton-Raphson (Diffrax)      1e-6/        
      SP2     ...             15         ...                           1e-6/        
      SP3     ...             30         ...                           1e-6/        
      SP4     ...             50         ...                           1e-6/        
      SP5     ...             100        ...                           1e-6/        
      WSP1    WingSailPlane   5          RK4                           27.34x1e-3   
      WSP2    ...             15         RK4                           6.62x1e-3    
      WSP3    ...             30         RK4                           2.49x1e-3    
      WSP4    ...             50         RK4                           1.27x1e-3    
      WSP5    ...             100        RK4                           0.575x1e-3   
      XRF1    XRF1 Airbus     70         RK4                           0.005        [1](#Table2),Index=1
      XRF2    ...             70         RK4                           0.005        [1](#Table2),Index=2
      XRF3    ...             70         RK4                           0.005        [1](#Table2),Index=3
      XRF4    ...             70         RK4                           0.005        [1](#Table2),Index=4
      XRF5    ...             70         Dopri5 (Diffrax)              0.005        [1](#Table2),Index=2
      XRF6    ...             70         RK4                           0.02         [1](#Table2),Index=2
      XRF7    ...             70         Implicit Euler (Diffrax)      1e-5/0.02    [1](#Table2),Index=2
      XRF8    ...             70         Implicit Kvaerno3 (Diffrax)   1e-5/0.02    [1](#Table2),Index=2

## SailPlane

### Runs

``` {#SP .python}
SP_folder = feniax.PATH / "../examples/SailPlane"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "inputs"
inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                         'LWingInner'],
                            FuselageBack=['BottomTail',
                                          'Fin'],
                            RWingInner=['RWingOuter'],
                            RWingOuter=None,
                            LWingInner=['LWingOuter'],
                            LWingOuter=None,
                            BottomTail=['LHorizontalStabilizer',
                                        'RHorizontalStabilizer'],
                            RHorizontalStabilizer=None,
                            LHorizontalStabilizer=None,
                            Fin=None
                            )

inp.fem.folder = pathlib.Path(SP_folder / 'FEM/')
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                           tolerance=1e-9)
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]

inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                      2e5,
                                                      2.5e5,
                                                      3.e5,
                                                      4.e5,
                                                      4.8e5,
                                                      5.3e5],
                                                     [0.,
                                                      2e5,
                                                      2.5e5,
                                                      3.e5,
                                                      4.e5,
                                                      4.8e5,
                                                      5.3e5]
                                                     ]
inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]
```

1.  SP1

    ``` {#SP1 .python}

    SP_folder = feniax.PATH / "../examples/SailPlane"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.eig_type = "inputs"
    inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                             'LWingInner'],
                                FuselageBack=['BottomTail',
                                              'Fin'],
                                RWingInner=['RWingOuter'],
                                RWingOuter=None,
                                LWingInner=['LWingOuter'],
                                LWingOuter=None,
                                BottomTail=['LHorizontalStabilizer',
                                            'RHorizontalStabilizer'],
                                RHorizontalStabilizer=None,
                                LHorizontalStabilizer=None,
                                Fin=None
                                )

    inp.fem.folder = pathlib.Path(SP_folder / 'FEM/')
    inp.fem.num_modes = 50
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "static"
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "newton"
    inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                               atol=1e-6,
                                               max_steps=50,
                                               norm="linalg_norm",
                                               kappa=0.01)
    # inp.systems.sett.s1.solver_library = "scipy"
    # inp.systems.sett.s1.solver_function = "root"
    # inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
    #                                           tolerance=1e-9)
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]

    inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5],
                                                         [0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5]
                                                         ]
    inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]

    inp.fem.num_modes = 5
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")
    run(inp, label=name)
    ```

2.  SP2

    ``` {#SP2 .python}

    SP_folder = feniax.PATH / "../examples/SailPlane"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.eig_type = "inputs"
    inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                             'LWingInner'],
                                FuselageBack=['BottomTail',
                                              'Fin'],
                                RWingInner=['RWingOuter'],
                                RWingOuter=None,
                                LWingInner=['LWingOuter'],
                                LWingOuter=None,
                                BottomTail=['LHorizontalStabilizer',
                                            'RHorizontalStabilizer'],
                                RHorizontalStabilizer=None,
                                LHorizontalStabilizer=None,
                                Fin=None
                                )

    inp.fem.folder = pathlib.Path(SP_folder / 'FEM/')
    inp.fem.num_modes = 50
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "static"
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "newton"
    inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                               atol=1e-6,
                                               max_steps=50,
                                               norm="linalg_norm",
                                               kappa=0.01)
    # inp.systems.sett.s1.solver_library = "scipy"
    # inp.systems.sett.s1.solver_function = "root"
    # inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
    #                                           tolerance=1e-9)
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]

    inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5],
                                                         [0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5]
                                                         ]
    inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]

    inp.fem.num_modes = 15
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")
    run(inp, label=name)
    ```

3.  SP3

    ``` {#SP3 .python}

    SP_folder = feniax.PATH / "../examples/SailPlane"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.eig_type = "inputs"
    inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                             'LWingInner'],
                                FuselageBack=['BottomTail',
                                              'Fin'],
                                RWingInner=['RWingOuter'],
                                RWingOuter=None,
                                LWingInner=['LWingOuter'],
                                LWingOuter=None,
                                BottomTail=['LHorizontalStabilizer',
                                            'RHorizontalStabilizer'],
                                RHorizontalStabilizer=None,
                                LHorizontalStabilizer=None,
                                Fin=None
                                )

    inp.fem.folder = pathlib.Path(SP_folder / 'FEM/')
    inp.fem.num_modes = 50
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "static"
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "newton"
    inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                               atol=1e-6,
                                               max_steps=50,
                                               norm="linalg_norm",
                                               kappa=0.01)
    # inp.systems.sett.s1.solver_library = "scipy"
    # inp.systems.sett.s1.solver_function = "root"
    # inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
    #                                           tolerance=1e-9)
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]

    inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5],
                                                         [0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5]
                                                         ]
    inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]

    inp.fem.num_modes = 30
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")
    run(inp, label=name)
    ```

4.  SP4

    ``` {#SP4 .python}

    SP_folder = feniax.PATH / "../examples/SailPlane"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.eig_type = "inputs"
    inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                             'LWingInner'],
                                FuselageBack=['BottomTail',
                                              'Fin'],
                                RWingInner=['RWingOuter'],
                                RWingOuter=None,
                                LWingInner=['LWingOuter'],
                                LWingOuter=None,
                                BottomTail=['LHorizontalStabilizer',
                                            'RHorizontalStabilizer'],
                                RHorizontalStabilizer=None,
                                LHorizontalStabilizer=None,
                                Fin=None
                                )

    inp.fem.folder = pathlib.Path(SP_folder / 'FEM/')
    inp.fem.num_modes = 50
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "static"
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "newton"
    inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                               atol=1e-6,
                                               max_steps=50,
                                               norm="linalg_norm",
                                               kappa=0.01)
    # inp.systems.sett.s1.solver_library = "scipy"
    # inp.systems.sett.s1.solver_function = "root"
    # inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
    #                                           tolerance=1e-9)
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]

    inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5],
                                                         [0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5]
                                                         ]
    inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]

    inp.fem.num_modes = 50
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")
    run(inp, label=name)
    ```

5.  SP5

    ``` {#SP5 .python}

    SP_folder = feniax.PATH / "../examples/SailPlane"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.eig_type = "inputs"
    inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                             'LWingInner'],
                                FuselageBack=['BottomTail',
                                              'Fin'],
                                RWingInner=['RWingOuter'],
                                RWingOuter=None,
                                LWingInner=['LWingOuter'],
                                LWingOuter=None,
                                BottomTail=['LHorizontalStabilizer',
                                            'RHorizontalStabilizer'],
                                RHorizontalStabilizer=None,
                                LHorizontalStabilizer=None,
                                Fin=None
                                )

    inp.fem.folder = pathlib.Path(SP_folder / 'FEM/')
    inp.fem.num_modes = 50
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "static"
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "newton"
    inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                               atol=1e-6,
                                               max_steps=50,
                                               norm="linalg_norm",
                                               kappa=0.01)
    # inp.systems.sett.s1.solver_library = "scipy"
    # inp.systems.sett.s1.solver_function = "root"
    # inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
    #                                           tolerance=1e-9)
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]

    inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5],
                                                         [0.,
                                                          2e5,
                                                          2.5e5,
                                                          3.e5,
                                                          4.e5,
                                                          4.8e5,
                                                          5.3e5]
                                                         ]
    inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]

    inp.fem.num_modes = 100
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")
    run(inp, label=name)
    ```

## wingSP

### Runs

``` {#wingSP .python}

wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
```

``` {#wingSP_dts .python}

wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
dts = [round(1./ eigenvals[i]**0.5, 2) for i in [5,15,30,50,100]]
```

1.  WSP1

    ``` {#WSP1 .python}


    wingSP_folder = feniax.PATH / "../examples/wingSP"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'c1': None}
    inp.fem.grid = "structuralGrid"
    inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
    eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
    inp.fem.eig_type = "inputs"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.t1 = 15.
    inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                                  [23, 2]]
    inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                         ]
    dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
    print(dts)
    inp.fem.num_modes = 5
    inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")

    run(inp, label=name)
    ```

2.  WSP2

    ``` {#WSP2 .python}


    wingSP_folder = feniax.PATH / "../examples/wingSP"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'c1': None}
    inp.fem.grid = "structuralGrid"
    inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
    eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
    inp.fem.eig_type = "inputs"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.t1 = 15.
    inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                                  [23, 2]]
    inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                         ]
    dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
    print(dts)
    inp.fem.num_modes = 15
    inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")

    run(inp, label=name)
    ```

3.  WSP3

    ``` {#WSP3 .python}


    wingSP_folder = feniax.PATH / "../examples/wingSP"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'c1': None}
    inp.fem.grid = "structuralGrid"
    inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
    eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
    inp.fem.eig_type = "inputs"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.t1 = 15.
    inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                                  [23, 2]]
    inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                         ]
    dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
    print(dts)
    inp.fem.num_modes = 30
    inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")

    run(inp, label=name)
    ```

4.  WSP4

    ``` {#WSP4 .python}


    wingSP_folder = feniax.PATH / "../examples/wingSP"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'c1': None}
    inp.fem.grid = "structuralGrid"
    inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
    eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
    inp.fem.eig_type = "inputs"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.t1 = 15.
    inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                                  [23, 2]]
    inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                         ]
    dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
    print(dts)
    inp.fem.num_modes = 50
    inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")

    run(inp, label=name)
    ```

5.  WSP4alpha05

    ``` {#WSP4alpha05 .python}


    wingSP_folder = feniax.PATH / "../examples/wingSP"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'c1': None}
    inp.fem.grid = "structuralGrid"
    inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
    eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
    inp.fem.eig_type = "inputs"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.t1 = 15.
    inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                                  [23, 2]]
    inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                         ]
    dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
    print(dts)
    inp.fem.num_modes = 50
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 0.5 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 0.5 * 6e5,  0., 0.]
                                                         ]
    inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")

    run(inp, label=name)
    ```

6.  WSP4alpha15

    ``` {#WSP4alpha15 .python}

    wingSP_folder = feniax.PATH / "../examples/wingSP"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'c1': None}
    inp.fem.grid = "structuralGrid"
    inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
    eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
    inp.fem.eig_type = "inputs"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.t1 = 15.
    inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                                  [23, 2]]
    inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                         ]
    dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
    print(dts)
    inp.fem.num_modes = 50
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1.5 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 1.5 * 6e5,  0., 0.]
                                                         ]
    inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")
    run(inp, label=name)
    ```

7.  WSP5

    ``` {#WSP5 .python}


    wingSP_folder = feniax.PATH / "../examples/wingSP"
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'c1': None}
    inp.fem.grid = "structuralGrid"
    inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
    eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
    inp.fem.eig_type = "inputs"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.t1 = 15.
    inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")#, max_steps=) #"rk4")
    inp.systems.sett.s1.solver_library = "diffrax"
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.xloads.follower_forces = True
    inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                                  [23, 2]]
    inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
    inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                         [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                         ]
    dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
    print(dts)
    inp.fem.num_modes = 100
    inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
    inp.driver.sol_path = pathlib.Path(
        f"./{name}")

    run(inp, label=name)
    ```

## XRF1

**WARNING: private model, not available open source**

Gust lengths and corresponding gust velocities that have been run here
and elsewhere. L~g~ 18.0,67.0,116.0,165.0,214 V0~g~
11.3047276743,14.0732311562,15.4214195361,16.3541764073,17.0785232867

  Index   Gust length \[m\]   Gust intensity   Intensity constant   u~inf~ \[m/s\]   rho~inf~ \[Kg/m^3008^\]   Mach
  ------- ------------------- ---------------- -------------------- ---------------- ------------------------- ------
  1       67                  14.0732311562    0.01                 200              1.225                     0.81
  2       67                  14.0732311562    2                    200              1.225                     0.81
  3       165\.               16.3541764073    0.01                 200              1.225                     0.81
  4       165\.               16.3541764073    2                    200              1.225                     0.81
  5       67                  14.0732311562    0.01                 200              1.225                     0\.
  6       67                  14.0732311562    2                    200              1.225                     0\.
  7       165\.               16.3541764073    0.01                 200              1.225                     0\.
  8       165\.               16.3541764073    2                    200              1.225                     0\.

  : Table with various gusts on the XRF1 that have been run in this work
  or in the past {#Table2}

``` {#XRF .python}
xrf1_folder = feniax.PATH / "../examples/XRF1/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{xrf1_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{xrf1_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{xrf1_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{xrf1_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{xrf1_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{xrf1_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{xrf1_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{xrf1_folder}/AERO/Control_nodes.npy"
```

### XRF1

``` {#XRF1 .python}

xrf1_folder = feniax.PATH / "../examples/XRF1/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{xrf1_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{xrf1_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{xrf1_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{xrf1_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{xrf1_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{xrf1_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{xrf1_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{xrf1_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*0.01
inp.systems.sett.s1.aero.gust.length = 67.
inp.systems.sett.s1.aero.gust.step = 0.5

run(inp, label=name)
```

### XRF2

``` {#XRF2 .python}

xrf1_folder = feniax.PATH / "../examples/XRF1/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{xrf1_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{xrf1_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{xrf1_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{xrf1_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{xrf1_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{xrf1_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{xrf1_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{xrf1_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.aero.gust.intensity = 16.3541764073 * 0.01
inp.systems.sett.s1.aero.gust.length = 165.
inp.systems.sett.s1.aero.gust.step = 0.05

run(inp, label=name)
```

### XRF3

``` {#XRF3 .python}

xrf1_folder = feniax.PATH / "../examples/XRF1/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{xrf1_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{xrf1_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{xrf1_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{xrf1_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{xrf1_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{xrf1_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{xrf1_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{xrf1_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*2 #11.304727674272842/10000
inp.systems.sett.s1.aero.gust.length = 67.
inp.systems.sett.s1.aero.gust.step = 0.5

run(inp, label=name)
```

### XRF4

``` {#XRF4 .python}

xrf1_folder = feniax.PATH / "../examples/XRF1/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{xrf1_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{xrf1_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{xrf1_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{xrf1_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{xrf1_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{xrf1_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{xrf1_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{xrf1_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.aero.gust.intensity = 16.3541764073*2 #11.304727674272842/10000
inp.systems.sett.s1.aero.gust.length = 165.
inp.systems.sett.s1.aero.gust.step = 0.5

run(inp, label=name)
```

### XRF5

``` {#XRF5 .python}

xrf1_folder = feniax.PATH / "../examples/XRF1/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{xrf1_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{xrf1_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{xrf1_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{xrf1_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{xrf1_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{xrf1_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{xrf1_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{xrf1_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5",#"Kvaerno3",
                                         )

inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*2 #11.304727674272842/10000
inp.systems.sett.s1.aero.gust.length = 67.
inp.systems.sett.s1.aero.gust.step = 0.5

run(inp, label=name)
```

### XRF6

``` {#XRF6 .python}

xrf1_folder = feniax.PATH / "../examples/XRF1/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{xrf1_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{xrf1_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{xrf1_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{xrf1_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{xrf1_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{xrf1_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{xrf1_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{xrf1_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")
inp.systems.sett.s1.tn = 501
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*2 #11.304727674272842/10000
inp.systems.sett.s1.aero.gust.length = 67.
inp.systems.sett.s1.aero.gust.step = 0.5

run(inp, label=name)
```

## Wrap up

``` python
save_times()
```
