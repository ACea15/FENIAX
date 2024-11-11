# [[file:modelgeneration.org::*Run][Run:1]]
import pathlib
#import jax.numpy as jnp
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_shardmain

label_gaf = "Dd1c7F1Scao-50"
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

inp.fem.grid = "structuralGridclamped"
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.eig_names = ["eigenvals_50.npy", "eigenvecs_50.npy"]
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./results1gust")
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "dynamic"
inp.system.t1 = 7.5
inp.system.tn = 4001
inp.system.solver_library = "runge_kutta"
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="rk4")
inp.system.xloads.modalaero_forces = True
inp.system.q0treatment = 2
inp.system.aero.c_ref = 3.0
inp.system.aero.u_inf = 180.
inp.system.aero.rho_inf = 1.
inp.system.aero.poles = f"./AERO/{Poles_file}.npy"
inp.system.aero.A = f"./AERO/{Ahh_file}.npy"
inp.system.aero.D = f"./AERO/{Dhj_file}.npy"
inp.system.aero.gust_profile = "mc"
#inp.system.aero.gust.intensity = 14.0732311562*0.001
inp.system.aero.gust.fixed_discretisation = [67, 180]
#inp.system.aero.gust.length = 67.
inp.system.aero.gust.step = 0.1
inp.system.aero.gust.shift = 0.
inp.system.aero.gust.panels_dihedral = "./AERO/Dihedral_d1c7.npy"
inp.system.aero.gust.collocation_points = "./AERO/Collocation_d1c7.npy"


inputflow = dict(length=[10,67],
                 intensity= [0.01,0.08, 0.2, 0.7]
                   )
inp.system.shard = dict(input_type="gust1",
                        inputs=inputflow)

sol = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)

# import jax
# import jax.numpy as jnp
# from functools import partial
# @partial(jax.jit, static_argnames=["a","d"])
# def r(a,d):
#     amin = min(a)
#     amax = max(a)
#     out = jnp.arange(amin,amax,d)
#     return out


# import jax
# import jax.numpy as jnp
# from jax import lax

# def compute_range_params(start, stop, step):
#     # Compute range parameters outside the jitted function
#     num_elements = (stop - start + step - 1) // step
#     return num_elements

# @jax.jit
# def dynamic_range_fn(start, step, num_elements):
#     # Convert num_elements to a static integer
#     num_elements = lax.convert_element_type(num_elements, jnp.int32)
    
#     # Use lax.iota to create a range
#     return start + lax.iota(jnp.int32, num_elements) * step

# # Example usage with precomputed num_elements
# start = 0
# stop = 10
# step = 2
# num_elements = compute_range_params(start, stop, step)
# result = dynamic_range_fn(start, step, num_elements)
# print(result)


