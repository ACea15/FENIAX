# [[file:modelgeneration.org::*Manoeuvre][Manoeuvre:1]]
import pathlib
import time
#import jax.numpy as jnp
import numpy as np
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_shardmain
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
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "static"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "newton"
inp.system.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.system.xloads.modalaero_forces = True
inp.system.xloads.x = [0.,1.]
inp.system.t = [1/3, 2/3, 1]#[0.25, 0.5, 0.75, 1]
inp.system.aero.c_ref = c_ref
inp.system.aero.u_inf = u_inf # a0(7000) =312
inp.system.aero.rho_inf = rho_inf
inp.system.aero.poles = f"./AERO/{Poles_file}.npy"
inp.system.aero.A = f"./AERO/{Ahh_file}.npy"
inp.system.aero.Q0_rigid = f"./AERO/Qhx{label_gaf}.npy"
inp.system.aero.qalpha = [[0.,  0., 0, 0, 0, 0],
                          [0.,  6 * np.pi / 180, 0, 0, 0, 0]] # interpolation: x=0 => qalpha=0
                                                              # x=1 => qalpha = 4   

#rho_rand = np.random.normal(0.6, 0.6*0.15, 500)
inp.driver.sol_path = pathlib.Path(
    "./results/manoeuvre1Shard")  
inputflow = dict(u_inf=np.linspace(190, 240, 8),
                 rho_inf=np.linspace(0.41, 0.81, 8))
inp.system.shard = dict(input_type="steadyalpha",
                        inputs=inputflow)
num_gpus = 8
solstatic1shard = feniax.feniax_shardmain.main(input_dict=inp, device_count=num_gpus)
# Manoeuvre:1 ends here
