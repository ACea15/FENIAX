import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor import solution
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

class TestRafaBeam:

    @pytest.fixture(scope="class")
    def sol(self):

        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = {'c1': None}
        inp.fem.folder = file_path / "../../../examples/RafaBeam/FEM/"
        inp.fem.num_modes = 100
        inp.fem.eig_type = "inputs"
        inp.driver.typeof = "intrinsic"
        inp.driver.save_fem = False
        inp.driver.sol_path=None
        inp.simulation.typeof = "single"
        inp.systems.sett.s1.solution = "dynamic"
        inp.systems.sett.s1.save = False
        inp.systems.sett.s1.t1 = 2.5
        inp.systems.sett.s1.tn = 2501
        inp.systems.sett.s1.solver_library = "runge_kutta" #"diffrax" #
        inp.systems.sett.s1.solver_function = "ode"
        inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
        inp.systems.sett.s1.init_states = dict(q1=["axial_parabolic",
                                                   ([0., 3., 3., 0., 0., 0], 20.)
                                                   ])
        config =  configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/RafaBeam"
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
