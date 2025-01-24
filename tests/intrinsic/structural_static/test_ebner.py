import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor import solution
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

@pytest.mark.legacy
class Test2PointDead:

    @pytest.fixture(scope="class")
    def solconf(self):
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = {'Beam1': ['Beam2'],
                                'Beam2': None}
        inp.fem.folder = file_path / "../../../examples/EbnerBeam/FEM"
        inp.fem.num_modes = 60
        inp.fem.eig_cutoff = 1e-4
        inp.fem.eig_type = "inputs"
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
        inp.systems.sett.s1.xloads.dead_points = [[9, 2], [18, 2]]
        inp.systems.sett.s1.xloads.x = [0, 1, 2]
        inp.systems.sett.s1.xloads.dead_interpolation = [[0, 0.85 / 2, 0.85],
                                                         [0, 1.35 / 2, 1.35]
                                                         ]
        inp.systems.sett.s1.t = [1, 2]
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol, config

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/EbnerBeam"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("StaticSystem", label="_s1")

        return sol.data
    
    def test_qs(self, solconf, data):

        sol, config = solconf
        assert jnp.allclose(sol.staticsystem_s1.q,
                            data.staticsystem_s1.q)

    def test_Xs(self, solconf, data):

        sol, config = solconf
        assert jnp.allclose(sol.staticsystem_s1.X2,
                            data.staticsystem_s1.X2,
                            atol=1e-5)
        assert jnp.allclose(sol.staticsystem_s1.X3,
                            data.staticsystem_s1.X3)

    def test_ra(self, solconf, data):

        sol, config = solconf
        assert jnp.allclose(-30.75,
                            (sol.staticsystem_s1.ra[-1,0,-1]
                             - config.fem.X[-1,0]),0.001)
        assert jnp.allclose(66.98,
                            (sol.staticsystem_s1.ra[-1,2,-1]
                             - config.fem.X[-1,2]),0.001)
        assert jnp.allclose(-8.12,
                            (sol.staticsystem_s1.ra[-1,0,9]
                             - config.fem.X[9,0]),0.001)
        assert jnp.allclose(24.83,
                            (sol.staticsystem_s1.ra[-1,2,9]
                             - config.fem.X[9,2]),0.001)
        
    def test_Cab(self, solconf, data):

        sol, config = solconf
        assert jnp.allclose(sol.staticsystem_s1.Cab,
                            data.staticsystem_s1.Cab)
