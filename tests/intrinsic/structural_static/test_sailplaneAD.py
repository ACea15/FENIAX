import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

class TestSailPlaneAD:

    @pytest.fixture(scope="class")
    def sol(self):

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
        inp.fem.folder = file_path / "../../../examples/SailPlane/FEM"
        inp.fem.num_modes = 20
        inp.driver.save_fem = False
        inp.driver.sol_path = None        
        inp.driver.typeof = "intrinsic"
        inp.simulation.typeof = "single"
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
        inp.system.save = False        
        inp.system.ad = dict(inputs=dict(t=1.5),
                             input_type="point_forces",
                             grad_type="jacrev",
                             objective_fun="var",
                             objective_var="ra",
                             objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                             )
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture(scope="class")
    def sol_epsilon(self):
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
        inp.fem.folder = file_path / "../../../examples/SailPlane/FEM"
        inp.fem.num_modes = 20
        inp.driver.save_fem = False
        inp.driver.sol_path = None        
        inp.driver.typeof = "intrinsic"
        inp.simulation.typeof = "single"
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
        inp.system.save = False
        # load at t = 1.5
        inp.system.t = [1]
        epsilon = 1e-4
        inp.system.ad = dict(inputs=dict(t=1.5 + epsilon),
                             input_type="point_forces",
                             grad_type="value",
                             objective_fun="var",
                             objective_var="ra",
                             objective_args=dict(t=(-1,), nodes=(25,), components=(2,))
                             )
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    
    def test_jac(self, sol):
        
        assert jnp.abs(0.705 - sol.staticsystem_sys1.jac['t']) < 0.001

    def test_jacfd(self, sol, sol_epsilon):
        
        epsilon = 1e-4
        jac_fd = (sol_epsilon.staticsystem_sys1.f_ad - sol.staticsystem_sys1.f_ad) / epsilon 
        assert jnp.abs(jac_fd - sol.staticsystem_sys1.jac['t']) / jnp.linalg.norm(jac_fd) < 1e-5

    
