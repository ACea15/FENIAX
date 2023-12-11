import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor import solution
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

class TestWingSP:

    @pytest.fixture(scope="class")
    def sol(self):

        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = {'c1': None}
        inp.fem.grid = "structuralGrid"
        inp.fem.folder = file_path / "../../../examples/wingSP/FEM/"
        inp.fem.num_modes = 50
        inp.fem.eig_type = "inputs"
        inp.driver.typeof = "intrinsic"
        inp.driver.save_fem = False
        inp.driver.sol_path = None
        inp.simulation.typeof = "single"
        inp.systems.sett.s1.solution = "dynamic"
        inp.systems.sett.s1.t1 = 15.
        inp.systems.sett.s1.tn = 15001
        inp.systems.sett.s1.save = False
        inp.systems.sett.s1.solver_library = "runge_kutta"
        inp.systems.sett.s1.solver_function = "ode"
        inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
        inp.systems.sett.s1.xloads.follower_forces = True
        inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                                      [23, 2]]
        inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
        inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                             [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                             ]
        config = configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/wingSP"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("DynamicSystem", label="_s1")

        return sol.data

    def test_qs(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_s1.q,
                            data.dynamicsystem_s1.q)

    def test_Xs(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_s1.X2,
                            data.dynamicsystem_s1.X2,
                            atol=1e-5)
        assert jnp.allclose(sol.dynamicsystem_s1.X3,
                            data.dynamicsystem_s1.X3)

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_s1.ra,
                            data.dynamicsystem_s1.ra)

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_s1.Cab,
                            data.dynamicsystem_s1.Cab)
