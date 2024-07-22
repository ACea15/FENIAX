import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor import solution
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

class TestBeamTipMoment:
    ...

class TestCurveFollower:

    @pytest.fixture(scope="class")
    def sol(self):
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = {'Beam1':None}
        inp.fem.folder = file_path / "../../../examples/Simo45Beam/FEM"
        inp.fem.num_modes = 90
        inp.fem.eig_type = "inputs"
        #inp.fem.fe_order_start = 1
        inp.driver.typeof = "intrinsic"
        inp.driver.save_fem = False
        inp.driver.sol_path = None
        inp.simulation.typeof = "single"
        inp.systems.sett.s1.solution = "static"
        inp.systems.sett.s1.solver_library = "diffrax"
        inp.systems.sett.s1.solver_function = "newton"
        inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                                   atol=1e-6,
                                                   max_steps=50,
                                                   norm="linalg_norm",
                                                   kappa=0.01)
        # inp.systems.sett.s1.solver_library = "scipy"
        # inp.systems.sett.s1.solver_function = "root"
        # inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
        #                                            tolerance=1e-9)
        #inp.systems.sett.s1.label = 'dq_001001'
        inp.systems.sett.s1.save = False                 
        inp.systems.sett.s1.xloads.follower_forces = True
        inp.systems.sett.s1.xloads.follower_points = [[15, 2]]
        inp.systems.sett.s1.xloads.x = [0, 1, 2, 3]
        inp.systems.sett.s1.xloads.follower_interpolation = [[0, 300, 600, 900]]
        inp.systems.sett.s1.t = [1, 2, 3]
        config =  configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/Simo45Follower"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("Modes")
        sol.load_container("Couplings")
        sol.load_container("StaticSystem", label="_s1")

        return sol.data

    
    def test_qs(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.q,
                            data.staticsystem_s1.q[:3])

    def test_Xs(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.X2,
                            data.staticsystem_s1.X2[:3],
                            atol=1e-5)
        assert jnp.allclose(sol.staticsystem_s1.X3,
                            data.staticsystem_s1.X3[:3])

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.ra,
                            data.staticsystem_s1.ra[:3])

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.Cab,
                            data.staticsystem_s1.Cab[:3])


class TestCurveDead:

    @pytest.fixture(scope="class")
    def sol(self):
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = {'Beam1':None}
        inp.fem.folder = file_path / "../../../examples/Simo45Beam/FEM"
        inp.fem.num_modes = 90
        inp.fem.eig_type = "inputs"
        #inp.fem.fe_order_start = 1
        inp.driver.typeof = "intrinsic"
        inp.driver.save_fem = False
        inp.driver.sol_path = None
        inp.simulation.typeof = "single"
        inp.systems.sett.s1.solution = "static"
        inp.systems.sett.s1.solver_library = "diffrax"
        inp.systems.sett.s1.solver_function = "newton"
        inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                                   atol=1e-6,
                                                   max_steps=50,
                                                   norm="linalg_norm",
                                                   kappa=0.01)
        #inp.systems.sett.s1.label = 'dq_00101'
        inp.systems.sett.s1.save = False 
        inp.systems.sett.s1.xloads.dead_forces = True
        inp.systems.sett.s1.xloads.dead_points = [[15, 2]]
        inp.systems.sett.s1.xloads.x = [0, 1, 2, 3]
        inp.systems.sett.s1.xloads.dead_interpolation = [[0, 300, 600, 900]]
        inp.systems.sett.s1.t = [1, 2, 3]
        config = configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/Simo45Dead"
        sol = solution.IntrinsicSolution(sol_path)
        #sol.load_container("Modes")
        #sol.load_container("Couplings")
        sol.load_container("StaticSystem", label="_s1")

        return sol.data

    
    def test_qs(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.q,
                            data.staticsystem_s1.q[:3])

    def test_Xs(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.X2,
                            data.staticsystem_s1.X2[:3],
                            atol=1e-5)
        assert jnp.allclose(sol.staticsystem_s1.X3,
                            data.staticsystem_s1.X3[:3])

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.ra,
                            data.staticsystem_s1.ra[:3])

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.Cab,
                            data.staticsystem_s1.Cab[:3])
