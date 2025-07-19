# [[file:modelgeneration.org::*Gust][Gust:2]]
import pathlib
import time

# import jax.numpy as jnp
import numpy as np
import jax.numpy as jnp
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_shardmain

label_dlm = "d1c7"
sol = "cao"
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

inp.fem.connectivity = dict(  # FusWing=['RWing',
    #          'LWing'],
    FusBack=["FusTail", "VTP"],
    FusFront=None,
    RWing=None,
    LWing=None,
    FusTail=None,
    VTP=["HTP", "VTPTail"],
    HTP=["RHTP", "LHTP"],
    VTPTail=None,
    RHTP=None,
    LHTP=None,
)
inp.fem.grid = f"./FEM/structuralGrid_{sol[:-1]}"
# inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.Ka_name = f"./FEM/Ka_{sol[:-1]}.npy"
inp.fem.Ma_name = f"./FEM/Ma_{sol[:-1]}.npy"
inp.fem.eig_names = [
    f"./FEM/eigenvals_{sol}{num_modes}.npy",
    f"./FEM/eigenvecs_{sol}{num_modes}.npy",
]
Ka = jnp.load(inp.fem.Ka_name)
Ma = jnp.load(inp.fem.Ma_name)
eigenvals = jnp.load(inp.fem.eig_names[0])
eigenvecs = jnp.load(inp.fem.eig_names[1])

inp.driver.typeof = "intrinsic"
inp.fem.num_modes = num_modes
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.system.name = "s1"
if sol[0] == "e":  # free model, otherwise clamped
    inp.system.bc1 = "free"
    inp.system.q0treatment = 1
inp.system.solution = "dynamic"
inp.system.t1 = 2
inp.system.tn = 2501
# inp.system.solver_library = "runge_kutta"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "ode"
#inp.system.solver_settings = dict(solver_name="rk4")
inp.system.solver_settings = dict(solver_name="Dopri5")
inp.system.xloads.modalaero_forces = True
inp.system.aero.c_ref = c_ref
inp.system.aero.u_inf = u_inf
inp.system.aero.rho_inf = rho_inf
inp.system.aero.poles = f"./AERO/{Poles_file}.npy"
inp.system.aero.A = f"./AERO/{Ahh_file}.npy"
inp.system.aero.D = f"./AERO/{Dhj_file}.npy"
inp.system.aero.gust_profile = "mc"
inp.system.aero.gust.intensity = 20
inp.system.aero.gust.length = 150.0
inp.system.aero.gust.step = 0.1
inp.system.aero.gust.shift = 0.0
inp.system.aero.gust.panels_dihedral = f"./AERO/Dihedral_{label_dlm}.npy"
inp.system.aero.gust.collocation_points = f"./AERO/Collocation_{label_dlm}.npy"

inp.driver.sol_path = pathlib.Path(f"./results/gust2_{sol}Shard")
inp.system.aero.gust.fixed_discretisation = [150, u_inf]
# Shard inputs
inputflow = dict(
    length=np.linspace(25, 265, 2),
    intensity=np.linspace(0.1, 3, 2),
    rho_inf=np.linspace(0.34, 0.48, 2),
)
#inp.system.operationalmode = "shardmap"
inp.system.shard = dict(input_type="gust1", inputs=inputflow)
inp.system.ad = dict(inputs=dict(eigenvals=eigenvals,
                                 eigenvecs=eigenvecs,
                                 Ka=Ka,
                                 Ma=Ma),
                     input_type="fem",
                     grad_type="jacrev",
                     objective_fun="pmax",
                     objective_var="X2",
                     objective_args=dict(nodes=(13,), components=(0,1,2,3,4,5))
                     )

num_gpus = 8
solgust21shard = feniax.feniax_shardmain.main(
    input_dict=inp, device_count=num_gpus, return_driver=True
)
# Gust:2 ends here
