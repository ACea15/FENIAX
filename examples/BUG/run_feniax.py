import pathlib
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import numpy as np
load=np.array([0.,2e4,2.5e4,3e4,4e4,4.8e4,5.3e4])
loading_grids=[2052,10002052]
fem_folder='./FEM/'
sol_path='./results_struct'
#def feniax_static(load,loading_grids,fem_folder='./FEM/',sol_path='./results_struct'):

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "inputs"

inp.fem.connectivity = dict(FusBack=['FusTail','VTP'],
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

inp.fem.folder = pathlib.Path(fem_folder)
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"

inp.driver.sol_path = pathlib.Path(sol_path)
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                          atol=1e-6,
                                          max_steps=50,
                                          norm="linalg_norm",
                                          kappa=0.01)
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[g,2] for g in loading_grids]

inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
force=list(load)
inp.systems.sett.s1.xloads.follower_interpolation = [force]*len(loading_grids)
inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]
config =  configuration.Config(inp)
sol = fem4inas.fem4inas_main.main(input_obj=config)
#return sol
