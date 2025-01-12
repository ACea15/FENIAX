import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

class TestWingSPad:

    @pytest.fixture(scope="class")
    def sol(self):

        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = {'c1': None}
        inp.fem.grid = "structuralGrid"
        inp.fem.folder = file_path / "../../../examples/wingSP/FEM/"        
        inp.fem.num_modes = 15
        inp.fem.eig_type = "inputs"
        inp.driver.typeof = "intrinsic"
        inp.driver.sol_path= None
        inp.driver.save_fem = False
        inp.simulation.typeof = "single"
        inp.system.solution = "dynamic"
        inp.system.t1 = 10.
        inp.system.tn = 1001
        inp.system.solver_library = "diffrax"
        inp.system.solver_function = "ode"
        inp.system.solver_settings = dict(solver_name="Dopri5")
        inp.system.xloads.follower_forces = True
        inp.system.xloads.follower_points = [[23, 0],
                                             [23, 2]]
        inp.system.xloads.x = [0, 4, 4+1e-6, 20]
        inp.system.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                    [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                    ]
        inp.system.save = False
        inp.system.ad = dict(inputs=dict(alpha=1.),
                             input_type="point_forces",
                             grad_type="jacrev",
                             objective_fun="max",
                             objective_var="X2",
                             objective_args=dict(nodes=(1,), components=(2,))
                             )
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture(scope="class")
    def sol_epsilon(self):

        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = {'c1': None}
        inp.fem.grid = "structuralGrid"
        inp.fem.folder = file_path / "../../../examples/wingSP/FEM/"                
        inp.fem.num_modes = 15
        inp.fem.eig_type = "inputs"
        inp.driver.typeof = "intrinsic"
        inp.driver.sol_path= None
        inp.driver.save_fem = False
        inp.simulation.typeof = "single"
        inp.system.solution = "dynamic"
        inp.system.t1 = 10.
        inp.system.tn = 1001
        inp.system.solver_library = "diffrax"
        inp.system.solver_function = "ode"
        inp.system.solver_settings = dict(solver_name="Dopri5")
        inp.system.xloads.follower_forces = True
        inp.system.xloads.follower_points = [[23, 0],
                                             [23, 2]]
        inp.system.xloads.x = [0, 4, 4+1e-6, 20]
        inp.system.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                    [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                    ]
        inp.system.save = False
        epsilon = 1e-4
        inp.system.ad = dict(inputs=dict(alpha=1. + epsilon),
                             input_type="point_forces",
                             grad_type="value",
                             objective_fun="max",
                             objective_var="X2",
                             objective_args=dict(nodes=(1,), components=(2,))
                             )
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol
    
    def test_jac(self, sol):
        
        assert jnp.abs(3726890.015 - sol.dynamicsystem_sys1.jac['alpha']) < 0.01

    def test_jacfd(self, sol, sol_epsilon):
        
        epsilon = 1e-4
        jac_fd = (sol_epsilon.dynamicsystem_sys1.f_ad - sol.dynamicsystem_sys1.f_ad) / epsilon 
        assert jnp.abs(jac_fd - sol.dynamicsystem_sys1.jac['alpha']) / jnp.linalg.norm(jac_fd) < 1e-4

    
