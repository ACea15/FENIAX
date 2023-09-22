import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp
import pickle
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

class TestBeamModal:

    @pytest.fixture(scope="class")
    def sol(self):

        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = [[]]
        inp.fem.folder = fem4inas.PATH / "examples/ArgyrisBeam/FEM"
        inp.fem.num_modes = 10
        inp.driver.typeof = "intrinsic"
        inp.simulation.typeof = "single"
        inp.ex.Cab_xtol = 1e-4
        config =  configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):

        sol_path = file_path / "data/argyris_beam_modal"
        with open(sol_path.with_suffix('pickle'), 'rb') as handle:
            solution = pickle.load(handle)
        return solution

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
        

class TestBeamSolution:

    @pytest.fixture(scope="class")
    def sol(self):

        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = [[]]
        inp.fem.folder = fem4inas.PATH / "examples/ArgyrisBeam/FEM"
        inp.fem.num_modes = 150
        inp.driver.typeof = "intrinsic"
        inp.simulation.typeof = "single"
        inp.ex.Cab_xtol = 1e-4
        inp.systems.sett.s1.solution = "static"
        inp.systems.sett.s1.solver_library = "diffrax"
        inp.systems.sett.s1.solver_function = "newton_raphson"
        inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                                   atol=1e-6,
                                                   max_steps=50,
                                                   norm=jnp.linalg.norm,
                                                   kappa=0.01)
        inp.systems.sett.s1.label = 'dq_0'
        inp.systems.sett.s1.xloads.follower_forces = True
        inp.systems.sett.s1.xloads.follower_points = [[25, 1]]
        inp.systems.sett.s1.xloads.follower_interpolation = [[
            [1., -3.7e3],
            [2, -12.1e3],
            [3, -17.5e3],
            [4, -39.3e3],
            [5, -61.0e3],
            [6, -94.5e3],
            [7, -120e3]
        ]]
        inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6, 7]
        config =  configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):

        sol_path = file_path / "data/argyris_beam_sol"
        with open(sol_path.with_suffix('pickle'), 'rb') as handle:
            solution = pickle.load(handle)
        return solution

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.staticsystems1.ra,
                            data.staticsystems1.ra,
                            atol=1e-4)

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.staticsystems1.Cab,
                            data.staticsystems1.Cab,
                            atol=1e-4)

class TestFrameModal:
    ...

class TestFrameSolution:
    ...

    
