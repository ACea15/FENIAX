# [[file:modelgeneration.org::*Run][Run:1]]
import pathlib
import jax.numpy as jnp
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

inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.eig_names = ["eigenvals_50.npy", "eigenvecs_50.npy"]
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./results1manoeuvre")
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
inp.system.t = [0.25, 0.5, 0.75, 1]
inp.system.aero.c_ref = 3.0
inp.system.aero.u_inf = 170.
inp.system.aero.rho_inf = 0.778
inp.system.aero.poles = f"./AERO/{Poles_file}.npy"
inp.system.aero.A = f"./AERO/{Ahh_file}.npy"
inp.system.aero.Q0_rigid = f"./AERO/Qhx{label_gaf}.npy"
inp.system.aero.qalpha = jnp.array(([0.,  0., 0, 0, 0, 0],
                                    [0.,  6 * jnp.pi / 180, 0, 0, 0, 0]))
inputflow = dict(u_inf=[100., 150., 170, 190.], rho_inf=[0.4, 0.6, 1., 1.2])
inp.system.shard = dict(input_type="steadyalpha",
                        inputs=inputflow)

# config =  configuration.Config(inp)
sol = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)
