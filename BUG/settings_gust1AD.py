# [[file:modelgeneration.org::*Run][Run:1]]
import pathlib
import time
import jax.numpy as jnp
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.reconstruction as reconstruction
import sys

if len(sys.argv) > 1:
    results_path = f"{sys.argv[1]}/results/"
else:
    results_path = "./results/"

label_dlm = "d1c7"
sol = "eao"
label_gaf = "Dd1c7F3Seao-100"
num_modes = 100
c_ref = 3.0
u_inf = 209.62786434059765
rho_inf = 0.41275511341689247
num_poles = 5
Dhj_file = f"D{label_gaf}p{num_poles}"
Ahh_file = f"A{label_gaf}p{num_poles}"
Poles_file = f"Poles{label_gaf}p{num_poles}"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "inputs"

inp.fem.connectivity = dict(# FusWing=['RWing',
                            #          'LWing'],
                            FusBack=['FusTail',
                                     'VTP'],
                            FusFront=None,
                            RWing=None,
                            LWing=None,
                            FusTail=None,
                            VTP=['HTP', 'VTPTail'],
                            HTP=['RHTP', 'LHTP'],
                            VTPTail=None,
                            RHTP=None,
                            LHTP=None,
                            )
inp.fem.grid = f"./FEM/structuralGrid_{sol[:-1]}"
#inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.Ka_name = f"./FEM/Ka_{sol[:-1]}.npy"
inp.fem.Ma_name = f"./FEM/Ma_{sol[:-1]}.npy"
inp.fem.eig_names = [f"./FEM/eigenvals_{sol}{num_modes}.npy",
                     f"./FEM/eigenvecs_{sol}{num_modes}.npy"]
inp.driver.typeof = "intrinsic"
inp.fem.num_modes = num_modes

inp.simulation.typeof = "single"
inp.system.name = "s1"
if sol[0] == "e": # free model, otherwise clamped
    inp.system.bc1 = 'free'
    inp.system.q0treatment = 1
inp.system.solution = "dynamic"
inp.system.t1 = 1.
inp.system.tn = 2501
inp.system.solver_library = "runge_kutta"
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="rk4")
inp.system.xloads.modalaero_forces = True
inp.system.aero.c_ref = c_ref
inp.system.aero.u_inf = u_inf
inp.system.aero.rho_inf = rho_inf
inp.system.aero.poles = f"./AERO/{Poles_file}.npy"
inp.system.aero.A = f"./AERO/{Ahh_file}.npy"
inp.system.aero.D = f"./AERO/{Dhj_file}.npy"
inp.system.aero.gust_profile = "mc"
inp.system.aero.gust.intensity = 30 #25
inp.system.aero.gust.length = 200. #150.
inp.system.aero.gust.step = 0.1
inp.system.aero.gust.shift = 0.
inp.system.aero.gust.panels_dihedral = f"./AERO/Dihedral_{label_dlm}.npy"
inp.system.aero.gust.collocation_points = f"./AERO/Collocation_{label_dlm}.npy"
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/gustAD_{sol}")
inp.system.ad = dict(inputs=dict(length = 110.,
                                 intensity = 20.,
                                 u_inf=u_inf,
                                 rho_inf=rho_inf),
                           input_type="gust1",
                           grad_type="jacrev",
                           objective_fun="max",
                           objective_var="X2",
                           objective_args=dict(nodes=(13,), components=(0,1,2,3,4,5))
                           )

config =  configuration.Config(inp)
sol1 = feniax.feniax_main.main(input_obj=config)
# Run:1 ends here

# [[file:modelgeneration.org::ad][ad]]
import pandas as pd
sol1.dynamicsystem_s1.objective
df =pd.DataFrame(dict(f=sol1.dynamicsystem_s1.objective.reshape(6))|{k:v.reshape(6) for k,v in sol1.dynamicsystem_s1.jac.items()}, index=range(6))
df
# ad ends here
