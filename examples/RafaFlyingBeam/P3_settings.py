# [[file:modelgen.org::*FENIAX][FENIAX:1]]
import feniax.preprocessor.configuration as configuration  
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pathlib
# FENIAX:1 ends here

# [[file:modelgen.org::*FENIAX][FENIAX:2]]
v_x = 1.
v_y = 0.
v_z = 0.
omega_x = 0.
omega_y = 1.
omega_z = 0.
gravity_forces = False
gravity_label = "g" if gravity_forces else ""
label = 'm8'
label_name = label + gravity_label
# FENIAX:2 ends here

# [[file:modelgen.org::*only-RB modes][only-RB modes:1]]
inp = Inputs()
# inp.log.level="debug"
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'rbeam': None, 'lbeam': None}
inp.fem.Ka_name = f"./FEM/Ka_{label}.npy"
inp.fem.Ma_name = f"./FEM/Ma_{label}.npy"
inp.fem.eig_names = [f"./FEM/eigenvalsX_{label}.npy",
                     f"./FEM/eigenvecsX_{label}.npy"]
inp.fem.grid = f"./FEM/structuralGrid_{label}"
inp.fem.num_modes = 6  
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.driver.sol_path= pathlib.Path(
    f"./results_{label_name}")
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "dynamic"
inp.system.bc1 = 'free'
inp.system.xloads.gravity_forces = gravity_forces
inp.system.t1 = 2.*10
inp.system.tn = 20000 * 10 + 1
inp.system.solver_library = "runge_kutta" #"diffrax" #
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="rk4")
inp.system.init_states = dict(q1=["nodal_prescribed",
                                  ([[v_x, v_y, v_z, omega_x, omega_y, omega_z],
                                    [v_x, v_y, v_z - omega_y * 1, omega_x, omega_y, omega_z],
                                    [v_x, v_y, v_z + omega_y * 1, omega_x, omega_y, omega_z]]
                                   ,)
                                  ]
                              )
config =  configuration.Config(inp)
sol = feniax.feniax_main.main(input_obj=config)

# all solution data in the sol object (everything are tensors)
# for instance: sol.dynamicsystem_sys1.ra position of node [time_step, component, node_id]
# sol.dynamicsystem_s1.X1 for velocities and so on
# only-RB modes:1 ends here

# [[file:modelgen.org::*Multiple cases][Multiple cases:1]]
RUN_MULTIPLE = True
if RUN_MULTIPLE:
    inp.fem.num_modes = 13 - 2
    vz = [0., 0.2, 0.3, 0.4, 0.5, 0.6]
    for i, vzi in enumerate(vz):
        label_i = label_name + f"vz{i}"
        inp.driver.sol_path= pathlib.Path(
            f"./results_sym{label_i}")
        inp.system.init_states = dict(q1=["nodal_prescribed",
                                          ([[v_x, v_y, v_z, omega_x, omega_y, omega_z],
                                            [v_x, v_y, v_z - omega_y * 1 + vzi, omega_x, omega_y, omega_z],
                                            [v_x, v_y, v_z + omega_y * 1 + vzi, omega_x, omega_y, omega_z]]
                                           ,)
                                          ]
                                      )
        config =  configuration.Config(inp)
        sol = feniax.feniax_main.main(input_obj=config)
  
        inp.driver.sol_path= pathlib.Path(
            f"./results_antisym{label_i}")
        inp.system.init_states = dict(q1=["nodal_prescribed",
                                          ([[v_x, v_y, v_z, omega_x, omega_y, omega_z],
                                            [v_x, v_y, v_z - omega_y * 1 - vzi, omega_x, omega_y, omega_z],
                                            [v_x, v_y, v_z + omega_y * 1 + vzi, omega_x, omega_y, omega_z]]
                                           ,)
                                          ]
                                      )
        config =  configuration.Config(inp)
        sol = feniax.feniax_main.main(input_obj=config)          
# all solution data in the sol object (everything are tensors)
# for instance: sol.dynamicsystem_sys1.ra position of node [time_step, component, node_id]
# sol.dynamicsystem_s1.X1 for velocities and so on
# Multiple cases:1 ends here
