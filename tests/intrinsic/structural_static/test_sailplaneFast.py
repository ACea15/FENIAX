import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor import solution
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

class TestSailPlaneFast:

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
        inp.fem.num_modes = 50
        inp.driver.fast_on = True
        inp.driver.typeof = "intrinsic"
        inp.driver.save_fem = False
        inp.driver.sol_path = None
        inp.simulation.typeof = "single"
        inp.system.operationalmode = "fast"
        inp.system.solution = "static"
        inp.system.save = False 
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
        inp.system.t = [1, 2, 3, 4, 5, 6]
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/SailPlane"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("Modes")
        sol.load_container("Couplings")
        sol.load_container("StaticSystem", label="_s1")

        return sol.data

    def test_phi1(self, sol, data):
        assert jnp.allclose(sol.modes.phi1, data.modes.phi1)

    def test_phi2(self, sol, data):

        assert jnp.allclose(sol.modes.phi2, data.modes.phi2)

    def test_psi1(self, sol, data):

        assert jnp.allclose(sol.modes.psi1, data.modes.psi1)

    def test_phi1l(self, sol, data):
        
        assert jnp.allclose(sol.modes.phi1l, data.modes.phi1l)

    def test_phi2l(self, sol, data):

        assert jnp.allclose(sol.modes.phi2l, data.modes.phi2l)

    def test_psi1l(self, sol, data):

        assert jnp.allclose(sol.modes.psi1l, data.modes.psi1l)
        
    def test_psi2l(self, sol, data):

        assert jnp.allclose(sol.modes.psi2l, data.modes.psi2l)

    def test_phi1ml(self, sol, data):

        assert jnp.allclose(sol.modes.phi1ml, data.modes.phi1ml)

    def test_gamma2(self, sol, data):

        assert jnp.allclose(sol.couplings.gamma2,
                            data.couplings.gamma2)
    
    def test_qs(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_sys1.q,
                            data.staticsystem_s1.q)

    def test_Xs(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_sys1.X2,
                            data.staticsystem_s1.X2,
                            atol=1e-5)
        assert jnp.allclose(sol.staticsystem_sys1.X3,
                            data.staticsystem_s1.X3)

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_sys1.ra,
                            data.staticsystem_s1.ra)

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_sys1.Cab,
                            data.staticsystem_s1.Cab)
