# IndustrialAC (clamped)

Verifying the nonlinear structural dynamics on a clamped configuration.

## Load modules

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
import pickle
import jax.numpy as jnp
import jax
import pandas as pd
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader
from tabulate import tabulate
```

## Run cases

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

  : Table with various gusts on the IndustrialAC that have been run in this work
  or in the past

``` {#industrialAC .python}
industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
```

### IndustrialAC

``` {#IndustrialAC .python}

industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
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

### industrialAC2

``` {#industrialAC2 .python}

industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
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

### industrialAC3

``` {#industrialAC3 .python}

industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
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

### industrialAC4

``` {#industrialAC4 .python}

industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
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

### industrialAC5

``` {#industrialAC5 .python}

industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
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

### industrialAC6

``` {#industrialAC6 .python}

industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
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

### industrialAC7

``` {#industrialAC7 .python}

industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

inp.systems.sett.s1.tn = 1501
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="ImplicitEuler",#"Kvaerno3",
          # stepsize_controller=dict(PIDController=dict(atol=1e-5,
              #                                            rtol=1e-5)),
          root_finder=dict(Newton=dict(atol=1e-5,
                                       rtol=1e-5))
                                         )

inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*2 #11.304727674272842/10000
inp.systems.sett.s1.aero.gust.length = 67.
inp.systems.sett.s1.aero.gust.step = 0.5

run(inp, label=name)
```

### industrialAC8

``` {#industrialAC8 .python}

industrialAC_folder = feniax.PATH / "../examples/IndustrialAC/"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "input_memory"
inp.fem.eigenvals = jnp.load(f"{industrialAC_folder}/FEM/Dreal70.npy")
inp.fem.eigenvecs = jnp.load(f"{industrialAC_folder}/FEM/Vreal70.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path(f"{industrialAC_folder}/FEM/")
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 70
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
mach = "081"
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.A = f"{industrialAC_folder}/AERO/AICs{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.D = f"{industrialAC_folder}/AERO/AICsQhj{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.aero.poles = f"{industrialAC_folder}/AERO/Poles{mach}_8r{inp.fem.num_modes}.npy"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.q0treatment = 2
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.shift = 0.
inp.systems.sett.s1.aero.gust.panels_dihedral = f"{industrialAC_folder}/AERO/Dihedral.npy"
inp.systems.sett.s1.aero.gust.collocation_points = f"{industrialAC_folder}/AERO/Control_nodes.npy"
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

inp.systems.sett.s1.tn = 1501
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="Kvaerno3", #"ImplicitEuler",#"Kvaerno3",
          # stepsize_controller=dict(PIDController=dict(atol=1e-5,
          #                                            rtol=1e-5)),
          root_finder=dict(Chord=dict(atol=1e-5,
                                      rtol=1e-5))                              
          # root_finder=dict(Newton=dict(atol=1e-6,
          #                              rtol=1e-6))
                                         )

inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*2 #11.304727674272842/10000
inp.systems.sett.s1.aero.gust.length = 67.
inp.systems.sett.s1.aero.gust.step = 0.5
run(inp, label=name)
```

``` python
save_times()
```

## Postprocessing

### Plotting functions

``` python
print(f"Format for figures: {figfmt}")
def fig_out(name, figformat=figfmt, update_layout=None):
    def inner_decorator(func):
        def inner(*args, **kwargs):
            fig = func(*args, **kwargs)
            if update_layout is not None:
                fig.update_layout(**update_layout)
            fig.show()
            figname = f"figs/{name}.{figformat}"
            fig.write_image(f"../{figname}", scale=6)
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
```

``` python

@fig_background
def industrialAC_wingtip2(sol1, sol2, dim, labels=None,nast_scale=None, nast_load=None):
    scale = 1./33.977
    fig=None
    x1, y1 = putils.pickIntrinsic2D(sol1.data.dynamicsystem_s1.t,
                                  sol1.data.dynamicsystem_s1.ra,
                                  fixaxis2=dict(node=150, dim=dim))
    x2, y2 = putils.pickIntrinsic2D(sol2.data.dynamicsystem_s1.t,
                                  sol2.data.dynamicsystem_s1.ra,
                                  fixaxis2=dict(node=150, dim=dim))

    fig = uplotly.lines2d(x1[:], (y1[:]-y1[0])*scale, fig,
                          dict(name=f"NMROM-G{labels[0]}",
                               line=dict(color="orange")
                               ))
    fig = uplotly.lines2d(x2[1:], (y2[:-1]-y2[0])*scale, fig,
                          dict(name=f"NMROM-G{labels[1]}",
                               line=dict(color="steelblue")
                               ))

    if nast_scale is not None:
        offset = 0. #u111m[nast_load[0],0,-1, dim]
        fig = uplotly.lines2d(t111m[nast_load[0]], (u111m[nast_load[0],:,-1, dim] - offset)*nast_scale*scale, fig,
                              dict(name=f"Lin. FE-G{labels[0]}",
                                   line=dict(color="black",
                                             dash="dash",
                                             width=1.5)
                                   ))
        offset2 = 0. #u111m[nast_load[1],0,-1, dim]
        fig = uplotly.lines2d(t111m[nast_load[1]], (u111m[nast_load[1],:,-1, dim] - offset2)*nast_scale*scale, fig,
                              dict(name=f"Lin. FE-G{labels[1]}",
                                   line=dict(color="red",
                                             dash="dot",
                                             width=1.5)
                                   ))
    dim_dict = {0:'x', 1:'y', 2:'z'}
    fig.update_yaxes(title=r'$\large u_%s / l$'%dim_dict[dim])
    fig.update_xaxes(range=[0, 4], title=r'$\large time \; [s]$')
    return fig

def subplots_wtips2(fun, *args, **kwargs):

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
    fig.update_layout(legend=dict(x=0.81, y=1))
    #fig.update_layout(showlegend=False,row=2, col=1)
    # fig.update_layout(showlegend=False,row=2, col=2)
    #fig.update_layout(fig1.layout)
    return fig



@fig_background
def industrialAC_wingtip4(sol1, sol2, sol3, sol4, dim, labels=None,nast_scale=None, nast_load=None):
    scale = 1./33.977
    fig=None
    x1, y1 = putils.pickIntrinsic2D(sol1.data.dynamicsystem_s1.t,
                                    sol1.data.dynamicsystem_s1.ra,
                                    fixaxis2=dict(node=150, dim=dim))
    x2, y2 = putils.pickIntrinsic2D(sol2.data.dynamicsystem_s1.t,
                                    sol2.data.dynamicsystem_s1.ra,
                                    fixaxis2=dict(node=150, dim=dim))
    x3, y3 = putils.pickIntrinsic2D(sol3.data.dynamicsystem_s1.t,
                                    sol3.data.dynamicsystem_s1.ra,
                                    fixaxis2=dict(node=150, dim=dim))
    x4, y4 = putils.pickIntrinsic2D(sol4.data.dynamicsystem_s1.t,
                                    sol4.data.dynamicsystem_s1.ra,
                                    fixaxis2=dict(node=150, dim=dim))

    fig = uplotly.lines2d(x1[1:], (y1[:-1]-y1[0])*scale, fig,
                          dict(name=f"NMROM-{labels[0]}",
                               line=dict(color="orange",
                                         dash="solid")
                               ))
    fig = uplotly.lines2d(x2[:], (y2[:]-y2[0])*scale, fig,
                          dict(name=f"NMROM-{labels[1]}",
                               line=dict(color="blue", dash="dot")
                               ))
    fig = uplotly.lines2d(x3[:], (y3[:]-y3[0])*scale, fig,
                          dict(name=f"NMROM-{labels[2]}",
                               line=dict(color="red")
                               ))
    fig = uplotly.lines2d(x4[:], (y4[:]-y4[0])*scale, fig,
                          dict(name=f"NMROM-{labels[3]}",
                               line=dict(color="grey", dash="dash")
                               ))

    dim_dict = {0:'x', 1:'y', 2:'z'}
    fig.update_yaxes(title=r'$\large u_%s / l$'%dim_dict[dim])
    fig.update_xaxes(range=[0, 4], title=r'$\large time \; [s]$')
    return fig
```

### Load Nastran data

``` python

# import pathlib
# import pickle
# import jax.numpy as jnp
# import jax
# import pandas as pd
# import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
# from feniax.preprocessor.inputs import Inputs
# import feniax.feniax_main
# import feniax.preprocessor.solution as solution
# import feniax.unastran.op2reader as op2reader
# from tabulate import tabulate
# 
examples_path = pathlib.Path("../../../../examples")

####### IndustrialAC ###########
nastran_path = examples_path / "IndustrialAC/NASTRAN/146-111/"
nas111 = op2reader.NastranReader(op2name=(nastran_path / "IndustrialAC-146run.op2"))
nas111.readModel()
t111, u111 = nas111.displacements()

nastran_pathm = examples_path / "IndustrialAC/NASTRAN/146-111_081"
nas111m = op2reader.NastranReader(op2name=(nastran_pathm / "IndustrialAC-146run.op2"))
nas111m.readModel()
t111m, u111m = nas111m.displacements()
```

### Aeroelastic dynamic loads on an industrial configuration

The studies presented in this section are based on a reference
configuration developed to industry standards known as IndustrialAC, which is
representative of a long-range wide-body transport airplane. The version
with a wing-tip extension in [@CEA2023] is employed to verify a gust
response against MSC Nastran linear solution of the full FE model. While
the previous results where purely structural, now the dynamic response
to an atmospheric disturbance or gust is computed. This aeroelastic
analysis is a requirement for certification purposes and it is one of
the main drivers in sizing the wings of high aspect ratio wings.
Furthermore, the previous examples showed the advantage of our approach
in terms of computational speed, but other than that results could be
obtained with commercial software. The geometrically nonlinear
aeroelastic response, however, it is not currently available in
commercial solutions that are bounded to linear analysis in the
frequency domain. Other research codes feature those additional physics,
yet are limited to simple models. Thus the added value in the proposed
approach comes at the intersection between the nonlinear physics arising
from large integrated displacements, computational efficiency and the
ability to enhance the models already built for industrial use.\
Fig. [1](#fig:industrialAC_modalshapes) shows the reference FE model with three
modal shapes. The FE model contains a total of around 177400 nodes,
which are condensed into 176 active nodes along the reference load axes
through interpolation elements. A Guyan or static condensation approach
is used for the reduction and the error in the natural frequencies
between full and reduced models is kept below 0.1% well beyond the 30th
mode. The aerodynamic model contains $\sim 1,500$ aerodynamic panels.
The simulations are carried out with a modal resolution of 70 modes and
a time step in the Runge-Kutta solver of 0.005.

```{=org}
#+name: fig:industrialAC_modalshapes
```
```{=org}
#+caption: Modified IndustrialAC reference configuration with characteristic modal shapes
```
```{=org}
#+attr_latex: :width 0.8\textwidth
```
[file:figs_ext/industrialAC_modalshapes3.pdf](figs_ext/industrialAC_modalshapes3.pdf)

1.  Linear response for low intensity gust

    A verification exercise is introduced first by applying two 1-cos
    gust shapes at a very low intensity, thus producing small
    deformations and a linear response. The flow Mach number is 0.81. A
    first gust is applied that we name as G1 of length 67 m and peak
    velocity 0.141 m/s, and a second gust, G2, of 165 m and peak
    velocity of 0.164 m/s. A snippet of the inputs to the simulation is
    display in Listing `\ref{code:dynamic}`{=latex}.

    ```{=latex}
    \begin{listing}[!ht]
    \begin{minted}[frame=single]{python}
    from feniax.preprocessor.inputs import Inputs
    \begin{minted}[frame=single]{python}
    inp.fem.folder = "./FEM/"
    inp.fem.num_modes = 70
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.t1 = 7.5
    inp.systems.sett.s1.tn = 2001
    inp.systems.sett.s1.solver_library = "runge_kutta"
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
    inp.systems.sett.s1.xloads.modalaero_forces = True
    inp.systems.sett.s1.aero.folder = "./AERO/"
    inp.systems.sett.s1.aero.c_ref = 7.271
    inp.systems.sett.s1.aero.u_inf = 200.
    inp.systems.sett.s1.aero.rho_inf = 1.225
    inp.systems.sett.s1.aero.gust_profile = "mc"
    inp.systems.sett.s1.aero.gust.intensity = 0.141
    inp.systems.sett.s1.aero.gust.length = 67.
    \end{minted}
    \caption{FENIAX of inputs for dynamic gust simulation}
    \label{code:dynamic}
    \end{listing}
    ```
    Fig. [1](#fig:GustXRF12) shows the normalised wing-tip response with
    our NMROM that accurately reproduces the linear solution based on
    the full FE model.

    ``` {#GustIndustrialAC2 .python}
    sol1= solution.IntrinsicReader("./IndustrialAC")
    sol2= solution.IntrinsicReader("./industrialAC2")
    fig, figname = fig_out(name)(subplots_wtips2)(industrialAC_wingtip2,sol1, sol2, labels=[1,2], nast_scale=0.01, nast_load=[2,6])
    figname
    ```

    ![Wing-tip response to low intensity
    gust](figs/GustXRF12.png){#fig:GustXRF12.png}

2.  Nonlinear response for high intensity gust

    The gust intensity in the previous section by a factor of 200 in
    order to show the effects of geometric nonlinearities that are only
    captured by the nonlinear solver. As seen in Fig.
    [2](#fig:GustXRF34), there are major differences in the $x$ and $y$
    components of the response due to follower and shortening effects,
    and a slight reduction in the $z$-component. These are well known
    geometrically nonlinear effects that are added to the analysis with
    no significant overhead.

    ``` {#GustindustrialAC34 .python}
    sol1= solution.IntrinsicReader("./industrialAC3")
    sol2= solution.IntrinsicReader("./industrialAC4")
    fig, figname = fig_out(name)(subplots_wtips2)(industrialAC_wingtip2, sol1, sol2, labels=[1,2], nast_scale=2., nast_load=[2,6])
    figname
    ```

    ![Wing-tip response to high intensity
    gust](figs/GustXRF34.png){#fig:GustXRF34}

    Snapshots of the 3D response are reconstructed for the G1 gust using
    the method verified above at the time points where tip displacement
    are maximum and minimum, i.e. 0.54 and 0.84 seconds. The front and
    side views together with the aircraft reference configuration are
    shown in Fig. [3](#fig:industrialACgust3D).

    ```{=org}
    #+name: fig:industrialACgust3D
    ```
    ```{=org}
    #+caption: 3D IndustrialAC Nonlinear gust response
    ```
    ```{=org}
    #+attr_latex: :width 1\textwidth
    ```
    [file:figs_ext/industrialACgust3D2.pdf](figs_ext/industrialACgust3D2.pdf)

    In previous examples the same Runge-Kutta 4 (RK4) time-marching
    scheme is used and now we explore the dynamic solution with other
    solvers to assess their accuracy and also their computational
    performance. Two explicit ODE solvers, RK4 and Dormand-Prince\'s 5/4
    method (labelled S1 and S2), and two implicit, Euler first order and
    Kvaerno\'s 3/2 method ((labelled S3 and S4)), are compared in Fig.
    [3](#fig:GustXRF3578). In order to justify the use of implicit
    solvers we reduce the time step from 0.005 to 0.02 seconds, at which
    point both explicit solvers diverge. Kvaerno\'s implicit solver
    remain stable and accurate despite the larger time step while the
    Euler implicit method is stable but do not yield accurate results.

    ``` {#GustXRF3578 .python}
    sol3= solution.IntrinsicReader("./industrialAC3")
    sol5= solution.IntrinsicReader("./industrialAC5")
    sol7= solution.IntrinsicReader("./industrialAC7")
    sol8= solution.IntrinsicReader("./industrialAC8")

    fig, figname = fig_out(name)(subplots_wtips2)(industrialAC_wingtip4, sol1=sol3, sol2=sol5, sol3=sol7, sol4=sol8,
                                                labels=["S1","S2","S3","S4"])
    figname
    ```

    ```{=org}
    #+name: fig:GustXRF3578
    ```
    ```{=org}
    #+caption: Wing-tip response to high intensity gust using implicit solvers
    ```
    ```{=org}
    #+attr_latex: :width 0.8\textwidth
    ```
    ```{=org}
    #+results: GustXRF3578
    ```
    [file:figs/GustXRF3578.pdf](figs/GustXRF3578.pdf)

    The computational times of the different solvers are shown in Table
    [2](#table:IndustrialAC_times). The implicit solvers have taken one order of
    magnitude more time to run despite the reduction in time step.
    Therefore the main take away this is that for moderately large
    frequency dynamics, the explicit solvers offer a much efficient
    solution. The turning point for using implicit solvers would be when
    the largest eigenvalue in Eqs. `\ref{eq2:sol_qs}`{=latex} led to
    prohibitly small time steps. In terms of the Nastran solution, we
    are not showing the whole simulation time because that would include
    the time to sample the DLM aerodynamics which are input into the
    NMROM as a post-processing step. Instead, the increase in time when
    adding an extra gust subcase to an already existing analysis is
    shown, i.e. the difference between one simulation that only computes
    one gust response and another with two. It is remarkable that the
    explicit solvers are faster on the nonlinear solution than the
    linear solution by a commercial software. Besides our highly
    efficient implementation, the main reason for this might be the
    Nastran solution involves first a frequency domain analysis and then
    an inverse Fourier transform to obtain the time-domain results.

    ``` {#IndustrialAC_times .python}
    dfruns = pd.read_csv('./run_times.csv',index_col=0).transpose()
    values = ["Time [s]"]
    values += [', '.join([str(round(dfruns[f'industrialAC{i}'].iloc[0], 2)) for i in [3,5,7,8]])]
    values += [0*60*60 + 1*60 + 21]
    header = ["NMROM [S1, S2, S3, S4]" ]
    header += ["$\Delta$ NASTRAN 146"]
    # df_sp = pd.DataFrame(dict(times=TIMES_DICT.values()),
    #                         index=TIMES_DICT.keys())

    # df_ = results_df['shift_conm2sLM25']
    # df_ = df_.rename(columns={"xlabel": "%Chord"})
    tabulate([values], headers=header, tablefmt='orgtbl')
    ```

                   NMROM \[S1, S2, S3, S4\]       $\Delta$ NASTRAN 146
      ------------ ------------------------------ ----------------------
      Time \[s\]   22.49, 18.94, 273.95, 847.89   81

      : Computational times IndustrialAC gust solution.

3.  Differentiation of aeroelastic response

    Similarly to the examples above, we now verify the AD implementation
    for the nonlinear aeroelastic response to the gust $G1$. The
    sensitivity of the six components of the wing root loads are
    computed with respect to the gust parameters $w_g$ and $L_g$, and
    the flow parameter $\rho_{\inf}$. The results are presented in
    [1](#table:IndustrialAC_AD). A very good agreement with the finite
    differences is found with $\epsilon=10^{-4}$.

    ```{=org}
    #+caption: Automatic differentiation in aeroelastic problem
    ```
    ```{=org}
    #+name: table:IndustrialAC_AD
    ```
    ```{=latex}
    \begin{table} [h!]
    \begin{center}
    \begin{tabular}{lllll}
    \toprule
     & $w_g$ & $L_g$ & $\rho_{\inf}$ \\
    \midrule
    $f_1$ (AD) & 12.180 & 6.666 & 477.208 \\
    $f_1$ (FD)  & 12.180 & 6.190  & 477.198  \\
    \hline
    $f_2 (AD)$ & 19.088 & 6.122 & 712.485 \\
    $f_2 (FD)$ & 19.088 & 7.045 & 712.514  \\
    \hline
    $f_3 (AD)$ & 65.574 & 8.218 & 1464.910 \\
    $f_3 (FD)$ & 65.574  & 7.813 & 1464.909  \\
    \hline
    $f_4 (AD)$ & 126.648& 21.598 & 2883.370 \\
    $f_4 (FD)$ & 126.648& 19.736  & 2883.371  \\
    \hline
    $f_5 (AD)$ & 330.759 & 85.224 & 5931.723 \\
    $f_5 (FD)$ & 330.759  & 97.188  & 5930.027  \\
    \hline
    $f_6$ (AD) & 252.128 & 48.423 & 7179.735 \\
    $f_6$ (FD) & 252.128  & 14.980  & 7180.023  \\
    \bottomrule
    \end{tabular}
    \end{center}
    \end{table}
    ```
