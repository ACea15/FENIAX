---
attr_latex: ':width 1.\\textwidth'
author:
- acea
caption: XRF1 flying at trim equilibrium perturbed by gust G1
name: 'fig:xrf1\_trimgust'
---

XRF1
====

Verifying the nonlinear structural dynamics on a clamped configuration.

Load modules
------------

``` {.python}
import plotly.express as px
import pyNastran.op4.op4 as op4
import matplotlib.pyplot as plt
import pdb
import datetime
import os
import shutil
REMOVE_RESULTS = False
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
import plotly.express as px
import pickle
import jax.numpy as jnp
import jax
import pandas as pd
import numpy as np
import pathlib
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.uplotly as uplotly
import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader
import feniax.plotools.nastranvtk.bdfdef as bdfdef
from tabulate import tabulate
examples_folder = pathlib.Path.cwd() / "../../../../examples"    
```

Run cases
---------

``` {.python}
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

``` {#xrf1trim1 .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
```

### XRF1trim-1~4g~

``` {#xrf1trim1_4g .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.systems.sett.s1.xloads.gravity = 9.807 * 4
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1.]
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/{name}")
run(inp, label=name)
```

### XRF1trimlin-1~4g~

``` {#xrf1trim1lin_4g .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.systems.sett.s1.xloads.gravity = 9.807 * 4
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1.]
inp.systems.sett.s1.nonlinear = -1
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/{name}")
run(inp, label=name)
```

### Trim1 + dynamic simulation of flying A/C

``` {#xrf1trim1_dyn .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.simulation.typeof = "serial"
inp.systems.sett.s1.xloads.gravity = 9.807 * 4
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1.]
inp.driver.sol_path = pathlib.Path(
    f"./{name}")
inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5", max_steps=int(5e4))#"rk4")
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.
inp.systems.sett.s2.dt = 5e-3
inp.systems.sett.s2.aero.poles = examples_folder / "XRF1trim" / f"NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
```

``` {#xrf1trim1_dynNl .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.simulation.typeof = "serial"
inp.systems.sett.s1.xloads.gravity = 9.807 * 4
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1.]
inp.driver.sol_path = pathlib.Path(
    f"./{name}")
inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5", max_steps=int(5e4))#"rk4")
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.
inp.systems.sett.s2.dt = 5e-3
inp.systems.sett.s2.aero.poles = examples_folder / "XRF1trim" / f"NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"

inp.systems.sett.s1.xloads.gravity = 9.807 * 4
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1.]
run(inp, label=name)
```

``` {#xrf1trim1_dynLin .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.simulation.typeof = "serial"
inp.systems.sett.s1.xloads.gravity = 9.807 * 4
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1.]
inp.driver.sol_path = pathlib.Path(
    f"./{name}")
inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5", max_steps=int(5e4))#"rk4")
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.
inp.systems.sett.s2.dt = 5e-3
inp.systems.sett.s2.aero.poles = examples_folder / "XRF1trim" / f"NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"

inp.systems.sett.s1.xloads.gravity = 9.807 * 4
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1.]
inp.systems.sett.s1.nonlinear = -1
inp.systems.sett.s2.nonlinear = -1
run(inp, label=name)
```

``` {#xrf1gust1_sett .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.simulation.typeof = "serial"
inp.systems.sett.s1.xloads.gravity = 9.807 * 1.
inp.systems.sett.s1.t = [1.]
inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5", max_steps=int(5e4))#"rk4")
inp.systems.sett.s2.xloads.modalaero_forces = True
inp.systems.sett.s2.xloads.gravity_forces = True
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.5
inp.systems.sett.s2.dt = 5e-3
inp.systems.sett.s2.aero.poles = examples_folder / "XRF1trim" / f"NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.D = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhj{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.gust.panels_dihedral = examples_folder / "XRF1trim" / "NASTRAN/AERO/Dihedral.npy"
inp.systems.sett.s2.aero.gust.collocation_points = examples_folder / "XRF1trim" / "NASTRAN/AERO/Control_nodes.npy"
```

``` {#xrf1gust1 .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.simulation.typeof = "serial"
inp.systems.sett.s1.xloads.gravity = 9.807 * 1.
inp.systems.sett.s1.t = [1.]
inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5", max_steps=int(5e4))#"rk4")
inp.systems.sett.s2.xloads.modalaero_forces = True
inp.systems.sett.s2.xloads.gravity_forces = True
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.5
inp.systems.sett.s2.dt = 5e-3
inp.systems.sett.s2.aero.poles = examples_folder / "XRF1trim" / f"NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.D = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhj{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.gust.panels_dihedral = examples_folder / "XRF1trim" / "NASTRAN/AERO/Dihedral.npy"
inp.systems.sett.s2.aero.gust.collocation_points = examples_folder / "XRF1trim" / "NASTRAN/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/{name}")
inp.systems.sett.s2.aero.gust_profile = "mc"
inp.systems.sett.s2.aero.gust.intensity = 28.14 #14.0732311562*2 #11.304727674272842/10000
inp.systems.sett.s2.aero.gust.length = 67.
inp.systems.sett.s2.aero.gust.step = 1.
inp.systems.sett.s2.aero.gust.shift = 0.
run(inp, label=name)
```

``` {#xrf1gust1lin .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.simulation.typeof = "serial"
inp.systems.sett.s1.xloads.gravity = 9.807 * 1.
inp.systems.sett.s1.t = [1.]
inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5", max_steps=int(5e4))#"rk4")
inp.systems.sett.s2.xloads.modalaero_forces = True
inp.systems.sett.s2.xloads.gravity_forces = True
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.5
inp.systems.sett.s2.dt = 5e-3
inp.systems.sett.s2.aero.poles = examples_folder / "XRF1trim" / f"NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.D = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhj{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.gust.panels_dihedral = examples_folder / "XRF1trim" / "NASTRAN/AERO/Dihedral.npy"
inp.systems.sett.s2.aero.gust.collocation_points = examples_folder / "XRF1trim" / "NASTRAN/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/{name}")
inp.systems.sett.s2.aero.gust_profile = "mc"
inp.systems.sett.s2.aero.gust.intensity = 28.14 #14.0732311562*2 #11.304727674272842/10000
inp.systems.sett.s2.aero.gust.length = 67.
inp.systems.sett.s2.aero.gust.step = 1.
inp.systems.sett.s2.aero.gust.shift = 0.
run(inp, label=name)
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/{name}")
inp.systems.sett.s1.nonlinear = -1
inp.systems.sett.s2.nonlinear = -1
run(inp, label=name)
```

``` {#xrf1gust2 .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.simulation.typeof = "serial"
inp.systems.sett.s1.xloads.gravity = 9.807 * 1.
inp.systems.sett.s1.t = [1.]
inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5", max_steps=int(5e4))#"rk4")
inp.systems.sett.s2.xloads.modalaero_forces = True
inp.systems.sett.s2.xloads.gravity_forces = True
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.5
inp.systems.sett.s2.dt = 5e-3
inp.systems.sett.s2.aero.poles = examples_folder / "XRF1trim" / f"NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.D = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhj{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.gust.panels_dihedral = examples_folder / "XRF1trim" / "NASTRAN/AERO/Dihedral.npy"
inp.systems.sett.s2.aero.gust.collocation_points = examples_folder / "XRF1trim" / "NASTRAN/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/{name}")
inp.systems.sett.s2.aero.gust_profile = "mc"
inp.systems.sett.s2.aero.gust.intensity = 28.14 #15.3541764073*2
inp.systems.sett.s2.aero.gust.length = 125.
inp.systems.sett.s2.aero.gust.step = 1.
inp.systems.sett.s2.aero.gust.shift = 0.
run(inp, label=name)
```

``` {#xrf1gust2lin .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.folder = examples_folder / 'XRF1trim/FEM/'
inp.fem.grid = "structuralGridc.txt"
inp.fem.eigenvals = jnp.load(inp.fem.folder / "Dreal100.npy")
inp.fem.eigenvecs = jnp.load(inp.fem.folder / "Vreal100.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.target = "trim"
inp.systems.sett.s1.bc1 = 'free'
# inp.systems.sett.s1.nonlinear = 
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.gravity_forces = True
#inp.systems.sett.s1.xloads.gravity = 0.5
# inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 180.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.A = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhh{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.B = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhx{inp.fem.num_modes}-000.npy"
inp.systems.sett.s1.aero.elevator_index = [-2, -1]
inp.systems.sett.s1.aero.elevator_link = [+1, -1]
inp.simulation.typeof = "serial"
inp.systems.sett.s1.xloads.gravity = 9.807 * 1.
inp.systems.sett.s1.t = [1.]
inp.systems.borrow = 's1'
inp.systems.sett.s2.solution = "dynamic"
inp.systems.sett.s2.target = "level"
inp.systems.sett.s2.bc1 = 'free'
inp.systems.sett.s2.q0treatment = 1
inp.systems.sett.s2.solver_library = "diffrax"#"runge_kutta"
inp.systems.sett.s2.solver_function = "ode"
inp.systems.sett.s2.solver_settings = dict(solver_name="Dopri5", max_steps=int(5e4))#"rk4")
inp.systems.sett.s2.xloads.modalaero_forces = True
inp.systems.sett.s2.xloads.gravity_forces = True
inp.systems.sett.s2.t = None
inp.systems.sett.s2.t1 = 1.5
inp.systems.sett.s2.dt = 5e-3
inp.systems.sett.s2.aero.poles = examples_folder / "XRF1trim" / f"NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.D = examples_folder / "XRF1trim" / f"NASTRAN/AERO/AICsQhj{inp.fem.num_modes}-000_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s2.aero.gust.panels_dihedral = examples_folder / "XRF1trim" / "NASTRAN/AERO/Dihedral.npy"
inp.systems.sett.s2.aero.gust.collocation_points = examples_folder / "XRF1trim" / "NASTRAN/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/{name}")
inp.systems.sett.s2.aero.gust_profile = "mc"
inp.systems.sett.s2.aero.gust.intensity = 28.14 #15.3541764073*2
inp.systems.sett.s2.aero.gust.length = 125.
inp.systems.sett.s2.aero.gust.step = 1.
inp.systems.sett.s2.aero.gust.shift = 0.
run(inp, label=name)
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/{name}")
inp.systems.sett.s1.nonlinear = -1
inp.systems.sett.s2.nonlinear = -1
run(inp, label=name)
```

``` {.python}
save_times()
```

Postprocessing
--------------

### Plotting functions

``` {.python}
scale_quality = 6
print(f"Format for figures: {figfmt}")
print(f"Image quality: {scale_quality}")  
def fig_out(name, figformat=figfmt, update_layout=None):
    def inner_decorator(func):
        def inner(*args, **kwargs):
            fig = func(*args, **kwargs)
            if update_layout is not None:
                fig.update_layout(**update_layout)
            fig.show()
            figname = f"figs/{name}.{figformat}"
            fig.write_image(f"../{figname}", scale=scale_quality)
            return fig, figname
        return inner
    return inner_decorator

def fig_background(func):

    def inner(*args, **kwargs):
        fig = func(*args, **kwargs)
        # if fig.data[0].showlegend is None:
        #     showlegend = True
        # else:
        #     showlegend = fig.data[0].showlegend

        fig.update_xaxes(
                       titlefont=dict(size=20),
                       tickfont = dict(size=20),
                       mirror=True,
                       ticks='outside',
                       showline=True,
                       linecolor='black',
            #zeroline=True,
        #zerolinewidth=2,
            #zerolinecolor='LightPink',
                       gridcolor='lightgrey')
        fig.update_yaxes(tickfont = dict(size=20),
                       titlefont=dict(size=20),
                       zeroline=True,
                       mirror=True,
                       ticks='outside',
                       showline=True,
                       linecolor='black',
                       gridcolor='lightgrey')
        fig.update_layout(plot_bgcolor='white',
                          yaxis=dict(zerolinecolor='lightgrey'),
                          showlegend=True, #showlegend,
                          margin=dict(
                              autoexpand=True,
                              l=0,
                              r=0,
                              t=2,
                              b=0
                          ))
        return fig
    return inner

@fig_background
def plot_Xtime(t, X,  dim, X0=[0.,0.,0., 0.,0.,0.], labels=None, node=150,
            scale=100./33.977, x_range=[0,4]):
    fig=None
    if labels is None:
        labels = list(range(X))
    colors = ["steelblue", "steelblue", "green", "green"]
    dashes = ["solid", "dash"]*2
    for i, Xi in enumerate(X):
        x1, y1 = putils.pickIntrinsic2D(t,
                                        Xi,
                                        fixaxis2=dict(node=node, dim=dim))
        if i == 0:
            y10=y1[0]
            print(y10)
        fig = uplotly.lines2d(x1, (y1 - X0[dim])/y10 * scale, fig,
                              dict(name=f"NMROM-{labels[i]}",
                                   line=dict(color=colors[i],
                                             dash=dashes[i])
                               ))

    dim_dict = {0:'x', 1:'y', 2:'z', 3:r'\theta_x', 4:r'\theta_y', 5:r'\theta_z'}      
    fig.update_yaxes(title=r'$\large u_{%s}$'%dim_dict[dim])
    fig.update_xaxes(range=x_range, title='time [s]')
    return fig

@fig_background
def plot_Xcomponents(X,  dim1, dim2, labels=None, node=150,
            scale1=1, scale2=1):
    fig=None
    if labels is None:
        labels = list(range(X))
    colors = ["steelblue", "steelblue", "green", "green"]
    dashes = ["solid", "dash"]*2
    fig = uplotly.lines2d([1], [1], fig,
                          dict(name=None,
                               showlegend=False,
                               #line=dict(color=colors[i]),
                               marker=dict(symbol="star", color="red",size=16)
                               ),
                          dict())

    for i, Xi in enumerate(X):
        x1, y1 = putils.pickIntrinsic2D(Xi,
                                        Xi,
                                        fixaxis1=dict(node=node, dim=dim1),
                                        fixaxis2=dict(node=node, dim=dim2)
                                        )
        if i == 0:
            x10 = x1[0]
            y10 = y1[0]
        if i == 2:
            x10 = x1[0]
            y10 = y1[0]

        fig = uplotly.lines2d(x1 / x10 * scale1, y1 / y10 * scale2, fig,
                              dict(name=f"NMROM-{labels[i]}",
                                   line=dict(color=colors[i],
                                             dash=dashes[i])
                               ))

    dim_dict = {0:r'\hat{F}_x', 1:'\hat{F}_y', 2:'\hat{F}_z',
                3:'\hat{M}_x', 4:'\hat{M}_y', 5:'\hat{M}_x'}
    fig.update_xaxes(title=r'$\large %s$' %dim_dict[dim1])
    fig.update_yaxes(title=r'$\large %s$' %dim_dict[dim2])
    return fig

def subplots_X(fun, *args, **kwargs):

    fig1 = fun(*args, dim=0, **kwargs)
    fig2 = fun(*args, dim=1, **kwargs)
    fig3 = fun(*args, dim=2, **kwargs)
    fig3.update_xaxes(title=None)
    fig2.update_xaxes(title=None)
    fig = make_subplots(rows=3, cols=1, horizontal_spacing=0.135, vertical_spacing=0.1,
                        # specs=[[{"colspan": 2}, None],
                        #       [{}, {}]]
                        )
    for i, f3i in enumerate(fig3.data):
        fig.add_trace(f3i,
                      row=1, col=1
                      )
    for i, f1i in enumerate(fig1.data):
        f1inew = f1i
        f1inew.showlegend = False          
        fig.add_trace(f1inew,
                      row=2, col=1
                      )
    for i, f2i in enumerate(fig2.data):
        f2inew = f2i
        f2inew.showlegend = False          
        fig.add_trace(f2inew,
                      row=3, col=1
                      )

    fig.update_xaxes(fig2.layout.xaxis,row=2, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig2.layout.yaxis,row=2, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_xaxes(fig1.layout.xaxis,row=3, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig1.layout.yaxis,row=3, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_xaxes(fig3.layout.xaxis,row=1, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig3.layout.yaxis,row=1, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_layout(plot_bgcolor='white',
                      yaxis=dict(zerolinecolor='lightgrey'),
                      showlegend=True, #showlegend,
                      margin=dict(
                          autoexpand=True,
                          l=0,
                          r=0,
                          t=2,
                          b=0
                          ))
    fig.update_layout(legend=dict(x=0.1, y=1))
    #fig.update_layout(showlegend=False,row=2, col=1)
    # fig.update_layout(showlegend=False,row=2, col=2)
    #fig.update_layout(fig1.layout)
    return fig

@fig_background
def plot_ffb_times(modes, y1, y2, label1, label2):

    fig = None
    fig = uplotly.lines2d(modes, y1, fig,
                              dict(name=label1,
                                   line=dict(color="blue")
                                   ),
                              dict())

    fig = uplotly.lines2d(modes, y2, fig,
                          dict(name=label2,
                               line=dict(color="red")
                               ),
                          dict())          
    fig.update_yaxes(type="log", exponentformat="power",
                     #tickformat= '.0e',
                     nticks=8)
    fig.update_layout(legend=dict(x=0.7, y=0.95),
                      height=650,
                      xaxis_title='Num. modes',
                      yaxis_title='Computational times [s]')
    return fig

@fig_background
def plot_ffb_error(modes, y1, label1):

    fig = None
    fig = uplotly.lines2d(modes, y1, fig,
                              dict(name=label1,
                                   line=dict(color="blue")
                                   ),
                              dict())
    fig.update_yaxes(type="log", exponentformat="power", #tickformat= '.0e',
                     nticks=8)
    fig.update_layout(showlegend=False,
                      #height=800,
                      xaxis_title='Num. modes',
                      yaxis_title='Cg error')
    return fig

def get_trimaoa_ti(ti, q, omega, phi1):
    q2= q[ti, 0:-1]
    q0i = - q2[2:]/ omega[2:]
    q0 = jnp.hstack([q2[:2], q0i])
    X0 = jnp.tensordot(phi1, q0, axes=(0, 0))
    return X0[4,0]

def get_trimaoa(q, omega, phi1):

    q_len = len(q)
    aoa = [get_trimaoa_ti(i,
                         q,
                         omega,
                         phi1) for i in range(q_len)]
    return jnp.array(aoa)

def get_trimelevator(q, ti=None):
    q_len = len(q)
    if ti is None:
        q_elevator = [q[i, -1] for i in range(q_len)]
    else:
        q_elevator= q[ti, -1]
    return jnp.array(q_elevator)

@fig_background
def plot_trimaoa(loads, alphas, labels):

    rad2grad = 180./3.141592
    colors = ['blue', 'green', 'red', 'black', 'orange', 'yellow']
    symbols = ["circle-open", "square", "diamond-open", "circle"]
    fig = None
    for i, (alphai, labeli) in enumerate(zip(alphas, labels)):
        fig = uplotly.lines2d(loads, rad2grad*alphai, fig,
                                  dict(name=labeli,
                                       line=dict(color=colors[i]),
                                       marker=dict(symbol=symbols[i], size=16)
                                       ),
                                  dict())

    #fig.update_yaxes(type="log", tickformat= '.0e', nticks=8)
    fig.update_layout(#showlegend=False,
                      #height=800,
                      yaxis_range=[0,19],
                      xaxis_range=[0.5,4.1],
                      legend=dict(x=0.1, y=0.9),
                      xaxis_title='Loads [n-g]',
                      yaxis_title=r'$Angle [^o]$')
    return fig
```

``` {.python}
@fig_background
def plot_Xtime(t, X,  dim, X0=[0.,0.,0., 0.,0.,0.], labels=None, node=150,
            scale=100./33.977, x_range=[0,4]):
    fig=None
    if labels is None:
        labels = list(range(X))
    colors = ["steelblue", "steelblue", "green", "green"]
    dashes = ["solid", "dash"]*2
    for i, Xi in enumerate(X):
        x1, y1 = putils.pickIntrinsic2D(t,
                                        Xi,
                                        fixaxis2=dict(node=node, dim=dim))
        if i == 0:
            y10=y1[0]
            print(y10)
        fig = uplotly.lines2d(x1, (y1 - X0[dim])/y10 * scale, fig,
                              dict(name=f"NMROM-{labels[i]}",
                                   line=dict(color=colors[i],
                                             dash=dashes[i])
                               ))

    dim_dict = {0:'x', 1:'y', 2:'z', 3:r'\theta_x', 4:r'\theta_y', 5:r'\theta_z'}      
    fig.update_yaxes(title=r'$\large u_{%s}$'%dim_dict[dim])
    fig.update_xaxes(range=x_range, title='time [s]')
    return fig

@fig_background
def plot_Xcomponents(X,  dim1, dim2, labels=None, node=150,
            scale1=1, scale2=1):
    fig=None
    if labels is None:
        labels = list(range(X))
    colors = ["steelblue", "steelblue", "green", "green"]
    dashes = ["solid", "dash"]*2
    fig = uplotly.lines2d([1], [1], fig,
                          dict(name=None,
                               showlegend=False,
                               #line=dict(color=colors[i]),
                               marker=dict(symbol="star", color="red",size=16)
                               ),
                          dict())

    for i, Xi in enumerate(X):
        x1, y1 = putils.pickIntrinsic2D(Xi,
                                        Xi,
                                        fixaxis1=dict(node=node, dim=dim1),
                                        fixaxis2=dict(node=node, dim=dim2)
                                        )
        if i == 0:
            x10 = x1[0]
            y10 = y1[0]
        if i == 2:
            x10 = x1[0]
            y10 = y1[0]

        fig = uplotly.lines2d(x1 / x10 * scale1, y1 / y10 * scale2, fig,
                              dict(name=f"NMROM-{labels[i]}",
                                   line=dict(color=colors[i],
                                             dash=dashes[i])
                               ))

    dim_dict = {0:r'\hat{F}_x', 1:'\hat{F}_y', 2:'\hat{F}_z',
                3:'\hat{M}_x', 4:'\hat{M}_y', 5:'\hat{M}_x'}
    fig.update_xaxes(title=r'$\large %s$' %dim_dict[dim1])
    fig.update_yaxes(title=r'$\large %s$' %dim_dict[dim2])
    return fig

def subplots_X(fun, *args, **kwargs):

    fig1 = fun(*args, dim=0, **kwargs)
    fig2 = fun(*args, dim=1, **kwargs)
    fig3 = fun(*args, dim=2, **kwargs)
    fig3.update_xaxes(title=None)
    fig2.update_xaxes(title=None)
    fig = make_subplots(rows=3, cols=1, horizontal_spacing=0.135, vertical_spacing=0.1,
                        # specs=[[{"colspan": 2}, None],
                        #       [{}, {}]]
                        )
    for i, f3i in enumerate(fig3.data):
        fig.add_trace(f3i,
                      row=1, col=1
                      )
    for i, f1i in enumerate(fig1.data):
        f1inew = f1i
        f1inew.showlegend = False          
        fig.add_trace(f1inew,
                      row=2, col=1
                      )
    for i, f2i in enumerate(fig2.data):
        f2inew = f2i
        f2inew.showlegend = False          
        fig.add_trace(f2inew,
                      row=3, col=1
                      )

    fig.update_xaxes(fig2.layout.xaxis,row=2, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig2.layout.yaxis,row=2, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_xaxes(fig1.layout.xaxis,row=3, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig1.layout.yaxis,row=3, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_xaxes(fig3.layout.xaxis,row=1, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig3.layout.yaxis,row=1, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_layout(plot_bgcolor='white',
                      yaxis=dict(zerolinecolor='lightgrey'),
                      showlegend=True, #showlegend,
                      margin=dict(
                          autoexpand=True,
                          l=0,
                          r=0,
                          t=2,
                          b=0
                          ))
    fig.update_layout(legend=dict(x=0.1, y=1))
    #fig.update_layout(showlegend=False,row=2, col=1)
    # fig.update_layout(showlegend=False,row=2, col=2)
    #fig.update_layout(fig1.layout)
    return fig

def get_trimaoa_ti(ti, q, omega, phi1):
    q2= q[ti, 0:-1]
    q0i = - q2[2:]/ omega[2:]
    q0 = jnp.hstack([q2[:2], q0i])
    X0 = jnp.tensordot(phi1, q0, axes=(0, 0))
    return X0[4,0]

def get_trimaoa(q, omega, phi1):

    q_len = len(q)
    aoa = [get_trimaoa_ti(i,
                         q,
                         omega,
                         phi1) for i in range(q_len)]
    return jnp.array(aoa)

def get_trimelevator(q, ti=None):
    q_len = len(q)
    if ti is None:
        q_elevator = [q[i, -1] for i in range(q_len)]
    else:
        q_elevator= q[ti, -1]
    return jnp.array(q_elevator)

@fig_background
def plot_trimaoa(loads, alphas, labels):

    rad2grad = 180./3.141592
    colors = ['blue', 'green', 'red', 'black', 'orange', 'yellow']
    symbols = ["circle-open", "square", "diamond-open", "circle"]
    fig = None
    for i, (alphai, labeli) in enumerate(zip(alphas, labels)):
        fig = uplotly.lines2d(loads, rad2grad*alphai, fig,
                                  dict(name=labeli,
                                       line=dict(color=colors[i]),
                                       marker=dict(symbol=symbols[i], size=16)
                                       ),
                                  dict())

    #fig.update_yaxes(type="log", tickformat= '.0e', nticks=8)
    fig.update_layout(#showlegend=False,
                      #height=800,
                      yaxis_range=[0,19],
                      xaxis_range=[0.5,4.1],
                      legend=dict(x=0.1, y=0.9),
                      xaxis_title='Loads [n-g]',
                      yaxis_title=r'$Angle [^o]$')
    return fig
```

### Aeroelastic dynamic loads on an industrial configuration

The studies presented in this section are based on a reference
configuration developed to industry standards known as XRF1, which is
representative of a long-range wide-body transport airplane. The version
with a wing-tip extension in [@CEA2023] is employed to verify a gust
response against NASTRAN linear solution. The FE model contains a total
of around 177400 nodes, which are condensed into 176 active nodes along
the reference load axes through interpolation elements. A Guyan or
static condensation approach is used for the reduction. The aerodynamic
model contains $\sim 1,500$ aerodynamic panels. The simulations are
carried out with a modal resolution of 70 modes. This aeroelastic
analysis is a requirement for certification purposes and it is one of
the main drivers in sizing the wings of high aspect ratio wings.

1.  Trim flight

    The calculation of the trim equilibrium is carried out for
    increasing values of the gravity acceleration as to replicate a
    pull-up manoeuvre. The aircraft 3D trim equilibrium is shown in Fig.
    for a range of pull-up manoeuvres from 1g to 3.5g. The reason to go
    beyond regulation requirements is to check the robustness of our
    solvers in large-deformations scenarios and also to appreciate
    better the differences between linear and nonlinear analysis.
    Moreover, by comparing with Nastran\'s 144 solution on the full FE
    model at small displacements, we verify our implementation of the
    trim. The airflow conditions are a density of $\rho = 0.778 Kg/m^3$
    and velocity $u_\infty = 180 m/s$. Shortening effects and follower
    forces become really significant as the gravity loading increases.

    [file:figs\_ext/xrf1trim.pdf](figs_ext/xrf1trim.pdf)

    Next in Fig. [1](#fig:xrf1_trimaoa) we show the angle of attack and
    elevator angles at trim obtained from our nonlinear solver and the
    linear Nastran solution. Small differences are found streaming from
    the nonlinearities and each particular solution process. For
    instance, the follower effect of aerodynamic forces mean they are
    less effective in counterbalancing the weight -which always points
    downwards- as larger deflections are present.

    ``` {#sol=xrf1trim1_4g .python}
    sol1= solution.IntrinsicReader("./xrf1trim1_4g")
    aoa = get_trimaoa(sol1.data.staticsystem_s1.q,
                      sol1.data.modes.omega,
                      sol1.data.modes.phi1)
    q_elevator = get_trimelevator(sol1.data.staticsystem_s1.q)
    ```

    ``` {#sol=xrf1trimlin1_4g .python}
    sol1lin= solution.IntrinsicReader("./xrf1trim1lin_4g")
    aoalin = get_trimaoa(sol1lin.data.staticsystem_s1.q,
                      sol1lin.data.modes.omega,
                      sol1lin.data.modes.phi1)
    qlin_elevator = get_trimelevator(sol1lin.data.staticsystem_s1.q)
    ```

    ``` {#sol=xrf1trim_nastran .python}
    aoa_nastran = jnp.array([7.552e-2,
                   1.511e-1,
                   2.266e-1,
                   3.021e-1,
                   ])
    qelevator_nastran = jnp.array([4.632e-2,
                         9.263e-2,
                         1.39e-1, 
                         1.853e-1
                         ])
    ```

    ``` {#sol=xrf1trim_plot .python}
    fig, figname = fig_out(name)(plot_trimaoa)(loads=[1,2,3,4],
                                               alphas=[aoa,
                                                       #aoalin,
                                                       aoa_nastran,
                                                       q_elevator,
                                                       #qlin_elevator,
                                                       qelevator_nastran],
                                               labels=["NLMROM-AoA",
                                                       #"LROM-AoA",
                                                       "NASTRAN-AoA",
                                                       "NLMROM-Elevator",
                                                       #"LROM-Elevator",
                                                       "NASTRAN-Elevator"])
    figname
    ```

    ![XRF1 trim elevator and angle of
    attack](figs/sol=xrf1trim_plot.png "xrf1_trimaoa")

    In addition to the static solution comparison, we verify our
    implementation by setting a dynamic simulation with gravity, angle
    of attack and the elevator angle from the static simulation as
    inputs. A good equilibrium state is attained with the vehicle flying
    straight over the course of 10 seconds dropping less that
    $10^{-4}$ m.

2.  Gust response

    Next we look at the gust response of the aircraft under trimmed
    conditions. Two high intensity 1-cos gusts are imposed on the 1-g
    trimmed flight: one of 67 m length, labelled G1, and another of 125
    m length, both with intensity of 28.14 m/s. Similar as in the trim
    case, this values are above regulations and we use them to push our
    solvers into the region of deformations where next generation of
    high-aspect ratio airplanes are expected to operate.

    ``` {#sol=xrf1gust_load .python}
    solxrf1gust1= solution.IntrinsicReader("./xrf1gust1")
    solxrf1gust1l= solution.IntrinsicReader("./xrf1gust1lin")
    solxrf1gust2= solution.IntrinsicReader("./xrf1gust2")
    solxrf1gust2l= solution.IntrinsicReader("./xrf1gust2lin")
    ```

    ``` {#sol=xrf1gust_tip .python}
    t = solxrf1gust1.data.dynamicsystem_s2.t
    ra = []
    ra.append(solxrf1gust1.data.dynamicsystem_s2.ra)
    ra.append(solxrf1gust1l.data.dynamicsystem_s2.ra)
    ra.append(solxrf1gust2.data.dynamicsystem_s2.ra)
    ra.append(solxrf1gust2l.data.dynamicsystem_s2.ra)
    labels = ["G1-NL", "G1-Lin", "G2-NL", "G2-Lin"]
    fig, figname = fig_out(name)(subplots_X)(plot_Xtime, t, ra, labels=labels, node=150, scale=1./33.977, x_range=[0,1.5])
    figname
    ```

    Fig. [2](#fig:xrf1gust_tip) presents the time evolution of the
    wing-tip normalised displacements and the differences between the
    linear and nonlinear analysis.

    [file:]()

    ``` {#sol=xrf1gust_root .python}
    t = solxrf1gust1.data.dynamicsystem_s2.t
    X2 = []
    X2.append(solxrf1gust1.data.dynamicsystem_s2.X2)
    X2.append(solxrf1gust1l.data.dynamicsystem_s2.X2)
    X2.append(solxrf1gust2.data.dynamicsystem_s2.X2)
    X2.append(solxrf1gust2l.data.dynamicsystem_s2.X2)
    labels = ["G1-NL", "G1-Lin", "G2-NL", "G2-Lin"]
    fig, figname = fig_out(name, update_layout=dict(yaxis_title=r"$\hat{X}_{2z}$",legend=dict(x=0.7, y=0.941), ))(plot_Xtime)(t, X2, dim=2, labels=labels, node=4, scale=1., x_range=[0,1.5])
    figname
    ```

    A more interesting metric to monitor is the loading at the root of
    the wing. As seen in Fig. [2](#fig:xrf1gust_root), higher loads are
    found from the nonlinear analysis. This is also reported in
    [cite:&CESNIK2014](cite:&CESNIK2014) and means that the dynamic
    loads from the linear analysis can be non-conservative. Safety
    factors are defined in part to account for those but their adequacy
    is not guaranteed in more flexible designs.

    ![Normalised wing vertical shear force time evolution with gust
    excitation](figs/sol=xrf1gust_root.png "xrf1gust_root")

    Fig. [3](#fig:xrf1gustcomp_root) shows the load diagram for the
    normalized shear and torsional forces acting at the wing-root. This
    case portraits the reason for building load envelopes with a large
    set of defining parameters: the sharper gust produces larger shear
    forces whereas the gust with a bigger length induces larger
    torsional moments. Similar behavior is seen for the bending-shear
    loads in Fig. [4](#fig:xrf1gustcomp_root2).

    ``` {#sol=xrf1gustcomp_root .python}
    t = solxrf1gust1.data.dynamicsystem_s2.t
    X2 = []
    X2.append(solxrf1gust1.data.dynamicsystem_s2.X2)
    X2.append(solxrf1gust1l.data.dynamicsystem_s2.X2)
    X2.append(solxrf1gust2.data.dynamicsystem_s2.X2)
    X2.append(solxrf1gust2l.data.dynamicsystem_s2.X2)
    labels = ["G1-NL", "G1-Lin", "G2-NL", "G2-Lin"]
    fig, figname = fig_out(name)(plot_Xcomponents)(X2, dim1=3, dim2=2, labels=labels, node=4, scale1=1, scale2=1)
    figname
    ```

    ![Wing root torsion-shear load diagram in response to two 1-cos
    gust](figs/sol=xrf1gustcomp_root.png "xrf1gustcomp_root")

    ``` {#sol=xrf1gustcomp_root2 .python}
    labels = ["G1-NL", "G1-Lin", "G2-NL", "G2-Lin"]
    fig, figname = fig_out(name)(plot_Xcomponents)(X2, dim1=4, dim2=2, labels=labels, node=4, scale1=1, scale2=1)
    figname
    ```

    ![Wing root bending-shear load diagram in response to two 1-cos
    gust](figs/sol=xrf1gustcomp_root2.png "xrf1gustcomp_root2")

    Finally the trim and peek of the gust states are displayed in Fig. .

    <file:figs_ext/xrf1gust1free.png>
