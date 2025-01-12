import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor import solution
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

class TestWingSPFast:

    @pytest.fixture(scope="class")
    def sol(self):

        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = {'c1': None}
        inp.fem.grid = "structuralGrid"
        inp.fem.folder = file_path / "../../../examples/wingSP/FEM/"
        inp.fem.num_modes = 50
        inp.fem.eig_type = "inputs"
        inp.driver.fast_on = True
        inp.driver.typeof = "intrinsic"
        inp.driver.save_fem = False
        inp.driver.sol_path = None
        inp.simulation.typeof = "single"
        inp.system.operationalmode = "fast"
        inp.system.solution = "dynamic"
        inp.system.t1 = 15.
        inp.system.tn = 15001
        inp.system.save = False
        inp.system.solver_library = "runge_kutta"
        inp.system.solver_function = "ode"
        inp.system.solver_settings = dict(solver_name="rk4")
        inp.system.xloads.follower_forces = True
        inp.system.xloads.follower_points = [[23, 0],
                                                      [23, 2]]
        inp.system.xloads.x = [0, 4, 4+1e-6, 20]
        inp.system.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                             [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                             ]
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/wingSP"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("DynamicSystem", label="_s1")

        return sol.data

    def test_qs(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.q,
                            data.dynamicsystem_s1.q)

    def test_Xs(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.X2,
                            data.dynamicsystem_s1.X2,
                            atol=1e-5)
        assert jnp.allclose(sol.dynamicsystem_sys1.X3,
                            data.dynamicsystem_s1.X3)

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.ra,
                            data.dynamicsystem_s1.ra)

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.Cab,
                            data.dynamicsystem_s1.Cab)
