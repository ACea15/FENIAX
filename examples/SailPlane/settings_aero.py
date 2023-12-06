import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import pyNastran.op4.op4 as op4

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
inp.fem.num_modes = 50
#inp.jax_np.allclose = 1e-5
Qhh = op4.read_op4(f"./NASTRAN/data_out/Qhh0_8-{inp.fem.num_modes}.op4")
Qhalpha = op4.read_op4(f"./NASTRAN/data_out/Qhx{inp.fem.num_modes}-0.8.op4")
Qhx = Qhalpha['Q_HX'][1][:,1:]
inp.driver.typeof = "intrinsic"
inp.driver.sol_path = pathlib.Path(
    f"./resultsAeroStatic_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton_raphson"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.aero.Qk_struct = [[0.], Qhh['Q_HH'][1][0].real]
inp.systems.sett.s1.aero.Q0_rigid = Qhx

# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                           tolerance=1e-9)
inp.systems.sett.s1.aero.qalpha = jnp.array([1,0,0,0,0]) * jnp.pi / 180
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.5
inp.systems.sett.s1.aero.c_ref = 2.
inp.systems.sett.s1.xloads.modalaero_forces = True

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)
