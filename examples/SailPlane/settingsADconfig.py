import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main

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

inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 20
inp.driver.typeof = "intrinsic"
inp.driver.ad_on = True
inp.driver.sol_path = pathlib.Path(
    f"./results_staticAD")
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "static"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "newton"
inp.system.solver_settings = dict(rtol=1e-6,
                                  atol=1e-6,
                                  max_steps=50,
                                  norm="linalg_norm",
                                  kappa=0.01)
inp.system.xloads.follower_forces = True
inp.system.xloads.follower_points = [[25, 2], [48, 2]]

inp.system.xloads.x = [0, 1, 2, 3, 4, 5, 6]
inp.system.xloads.follower_interpolation = [[0.,
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

# load at t = 1.5
inp.system.t = [1]
inp.system.ad = dict(inputs=dict(t=1.5),
                     input_type="point_forces",
                     grad_type="jacrev",
                     objective_fun="var",
                     objective_var="ra",
                     objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                     )
config =  configuration.Config(inp)
sol11 = feniax.feniax_main.main(input_obj=config)

epsilon = 1e-3
inp.system.ad = dict(inputs=dict(t=1.5+epsilon),
                     input_type="point_forces",
                     grad_type="jacrev",
                     objective_fun="var",
                     objective_var="ra",
                     objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                     )
config2 =  configuration.Config(inp)
sol12 = feniax.feniax_main.main(input_obj=config2)

jac1 = sol11.staticsystem_sys1.jac['t']
jac1_fd = (sol12.staticsystem_sys1.f_ad - sol11.staticsystem_sys1.f_ad) / epsilon
print(f"error jacs: {(jac1-jac1_fd)/jnp.linalg.norm(jac1)*100} %")

# load at t = 3.5
inp.system.t = [1, 2, 3]
inp.system.ad = dict(inputs=dict(t=3.5),
                     input_type="point_forces",
                     grad_type="jacrev",
                     objective_fun="var",
                     objective_var="ra",
                     objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                     )
config =  configuration.Config(inp)
sol21 = feniax.feniax_main.main(input_obj=config)

epsilon = 1e-3
inp.system.ad = dict(inputs=dict(t=3.5+epsilon),
                     input_type="point_forces",
                     grad_type="jacrev",
                     objective_fun="var",
                     objective_var="ra",
                     objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                     )
config2 =  configuration.Config(inp)
sol22 = feniax.feniax_main.main(input_obj=config2)

jac2 = sol21.staticsystem_sys1.jac['t']
jac2_fd = (sol22.staticsystem_sys1.f_ad - sol21.staticsystem_sys1.f_ad) / epsilon
print(f"error jacs2: {(jac2-jac2_fd)/jnp.linalg.norm(jac2)*100} %")

# load at t = 5.5
inp.system.t = [1, 2, 3, 4, 5]
inp.system.ad = dict(inputs=dict(t=5.5),
                     input_type="point_forces",
                     grad_type="jacrev",
                     objective_fun="var",
                     objective_var="ra",
                     objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                     )
config =  configuration.Config(inp)
sol31 = feniax.feniax_main.main(input_obj=config)

epsilon = 1e-3
inp.system.ad = dict(inputs=dict(t=5.5+epsilon),
                     input_type="point_forces",
                     grad_type="jacrev",
                     objective_fun="var",
                     objective_var="ra",
                     objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                     )
config2 =  configuration.Config(inp)
sol32 = feniax.feniax_main.main(input_obj=config2)

jac3 = sol31.staticsystem_sys1.jac['t']
jac3_fd = (sol32.staticsystem_sys1.f_ad - sol31.staticsystem_sys1.f_ad) / epsilon
print(f"error jacs3: {(jac3-jac3_fd)/jnp.linalg.norm(jac3)*100} %")

###########################
inp.system.t = [1, 2, 3, 4, 5]
inp.driver.sol_path = pathlib.Path(
    "./results_staticADfem")
inp.system.ad = dict(inputs=dict(Ka=jnp.load("./FEM/Ka.npy"),
                                 Ma=jnp.load("./FEM/Ma.npy"),
                                 eigenvals=jnp.load("./FEM/eigenvals.npy"),
                                 eigenvecs=jnp.load("./FEM/eigenvecs.npy")
                                 ),
                     input_type="fem",
                     grad_type="jacrev",
                     objective_fun="var",
                     objective_var="ra",
                     objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                     )
config =  configuration.Config(inp)
sol = feniax.feniax_main.main(input_obj=config)
sol.staticsystem_sys1.jac
