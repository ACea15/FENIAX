import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor import solution
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

@pytest.mark.private
class TestXRF1:

    @pytest.fixture(scope="class")
    def sol(self):
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.eig_type = "input_memory"
        inp.fem.eigenvals = jnp.load(feniax.PATH / "../examples/XRF1/FEM/Dreal70.npy")
        inp.fem.eigenvecs = jnp.load(feniax.PATH / "../examples/XRF1/FEM/Vreal70.npy").T
        inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                                [8], [9], [10, 11], [29], [12], [],
                                [14], [15], [16, 21], [17, 23, 25],
                                [18], [19], [20], [], [22], [], [24], [],
                                [26], [], [28], [], [30], [], []]
        inp.fem.folder = pathlib.Path(feniax.PATH / "../examples/XRF1/FEM")
        inp.driver.save_fem = False
        inp.fem.grid = "structuralGridc.txt"
        inp.fem.num_modes = 70
        inp.driver.typeof = "intrinsic"
        # inp.driver.sol_path = pathlib.Path(
        #     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        inp.driver.sol_path = None
        inp.simulation.typeof = "single"
        inp.systems.sett.s1.solution = "dynamic"
        inp.systems.sett.s1.save = False
        inp.systems.sett.s1.t1 = 15.
        inp.systems.sett.s1.tn = 3001
        inp.systems.sett.s1.solver_library = "runge_kutta"
        inp.systems.sett.s1.solver_function = "ode"
        inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
        inp.systems.sett.s1.xloads.modalaero_forces = True
        inp.systems.sett.s1.q0treatment = 2
        inp.systems.sett.s1.aero.c_ref = 7.271
        inp.systems.sett.s1.aero.u_inf = 200.
        inp.systems.sett.s1.aero.rho_inf = 1.225
        inp.systems.sett.s1.aero.A = feniax.PATH / f"../examples/XRF1/NASTRAN/AERO/AICs000_8r{inp.fem.num_modes}.npy"
        inp.systems.sett.s1.aero.D = feniax.PATH / f"../examples/XRF1/NASTRAN/AERO/AICsQhj000_8r{inp.fem.num_modes}.npy"
        inp.systems.sett.s1.aero.poles = feniax.PATH / f"../examples/XRF1/NASTRAN/AERO/Poles000_8r{inp.fem.num_modes}.npy"
        inp.systems.sett.s1.aero.gust_profile = "mc"
        inp.systems.sett.s1.aero.gust.intensity = 14.0732311562*3 #11.304727674272842/10000
        inp.systems.sett.s1.aero.gust.length = 67.
        inp.systems.sett.s1.aero.gust.step = 1.
        inp.systems.sett.s1.aero.gust.shift = 0.
        inp.systems.sett.s1.aero.gust.panels_dihedral = jnp.load(feniax.PATH / "../examples/XRF1/NASTRAN/AERO/Dihedral.npy")
        inp.systems.sett.s1.aero.gust.collocation_points = feniax.PATH / "../examples/XRF1/NASTRAN/AERO/Control_nodes.npy"
        
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/XRF1/gust1/"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("Modes")
        sol.load_container("Couplings")
        sol.load_container("DynamicSystem", label="_s1")

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

    def test_gamma1(self, sol, data):
        
        assert jnp.allclose(sol.couplings.gamma1,
                            data.couplings.gamma1)

    def test_gamma2(self, sol, data):

        assert jnp.allclose(sol.couplings.gamma2,
                            data.couplings.gamma2)
    
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
