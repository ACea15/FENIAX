import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor import solution
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import jax.numpy as jnp
import pytest
import pathlib

file_path = pathlib.Path(__file__).parent

# sol_path = "./data/ArgyrisBeam"
# sol = solution.IntrinsicSolution(sol_path)
# sol.load_container("Modes")
# sol.load_container("Couplings")
# sol.load_container("StaticSystem", label="_s1")
# sol_path2 = "./ArgyrisBeam"
# sol2 = solution.IntrinsicSolution(sol_path2)
# sol2.load_container("Modes")
# sol2.load_container("Couplings")
# sol2.load_container("StaticSystem", label="_s1")

class TestBeamModal:

    @pytest.fixture(scope="class")
    def sol_path(self):
        path = pathlib.Path(
            "./ArgyrisBeam")
        return path

    @pytest.fixture(scope="class")
    def sol(self, sol_path):

        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = [[]]
        inp.fem.folder = fem4inas.PATH / "../examples/ArgyrisBeam/FEM"
        inp.fem.num_modes = 150
        inp.fem.eig_type = "inputs"
        inp.driver.typeof = "intrinsic"
        # inp.driver.sol_path = pathlib.Path(
        #             "./ArgyrisBeamModal")
        inp.driver.save_presimulation = False
        inp.simulation.typeof = "single"
        config = configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)

        return obj_sol

    @pytest.fixture
    def data(self):

        sol_path = file_path / "data/ArgyrisBeam"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("Modes")
        sol.load_container("Couplings")
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

class TestBeamSolution:

    @pytest.fixture(scope="class")
    def sol(self):
        
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = [[]]
        inp.fem.folder = fem4inas.PATH / "../examples/ArgyrisBeam/FEM"
        inp.fem.num_modes = 150
        inp.fem.eig_type = "inputs"
        inp.driver.typeof = "intrinsic"
        inp.driver.sol_path = pathlib.Path(
                    "./ArgyrisBeam")
        inp.driver.save_presimulation = False
        inp.simulation.typeof = "single"
        inp.systems.sett.s1.solution = "static"
        inp.systems.sett.s1.save = False 
        inp.systems.sett.s1.solver_library = "diffrax"
        inp.systems.sett.s1.solver_function = "newton_raphson"
        inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                                   atol=1e-6,
                                                   max_steps=50,
                                                   norm=jnp.linalg.norm,
                                                   kappa=0.01)
        inp.systems.sett.s1.label = 'dq_001001'
        inp.systems.sett.s1.xloads.follower_forces = True
        inp.systems.sett.s1.xloads.follower_points = [[25, 1]]
        inp.systems.sett.s1.xloads.x = [0, 1, 2]
        inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                             -3.7e3,
                                                             -12.1e3
                                                              ]
                                                             ]
        inp.systems.sett.s1.t = [1, 2]
        config = configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)

        return obj_sol

    @pytest.fixture
    def data(self):

        sol_path = file_path / "data/ArgyrisBeam"
        sol = solution.IntrinsicSolution(sol_path) #solution.IntrinsicSolution(sol_path)
        sol.load_container("Modes")
        sol.load_container("Couplings")
        sol.load_container("StaticSystem", label="_s1")

        return sol.data

    def test_qs(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.q,
                            data.staticsystem_s1.q[:2],
                            atol=1e-5)

    def test_Xs(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.X2,
                            data.staticsystem_s1.X2[:2])
        assert jnp.allclose(sol.staticsystem_s1.X3,
                            data.staticsystem_s1.X3[:2])

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.ra,
                            data.staticsystem_s1.ra[:2],
                            atol=1e-5)

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.staticsystem_s1.Cab,
                            data.staticsystem_s1.Cab[:2])

class TestFrameModal:

    @pytest.fixture(scope="class")
    def sol_path(self):
        path = pathlib.Path(
            "./ArgyrisFrameModal")
        return path

    @pytest.fixture(scope="class")
    def sol(self, sol_path):
        
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = [[1], []]
        inp.fem.folder = fem4inas.PATH / "../examples/ArgyrisFrame/FEM"
        inp.fem.eig_type = "inputs"
        inp.fem.num_modes = 120
        inp.fem.fe_order_start = 1
        inp.driver.typeof = "intrinsic"
        inp.driver.sol_path = sol_path
        inp.simulation.typeof = "single"
        config = configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)

        return obj_sol

    @pytest.fixture
    def data(self):

        sol_path = file_path / "data/ArgyrisFrame"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("Modes")
        sol.load_container("Couplings")
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

class TestFrameSolution:

    @pytest.fixture(scope="class")
    def sol(self):
        
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.connectivity = [[1], []]
        inp.fem.folder = fem4inas.PATH / "../examples/ArgyrisFrame/FEM"
        inp.fem.num_modes = 120
        inp.fem.fe_order_start = 1
        inp.fem.eig_type = "inputs"
        inp.driver.typeof = "intrinsic"
        inp.simulation.typeof = "single"
        inp.systems.sett.s1.solution = "static"
        inp.systems.sett.s1.save = False 
        inp.systems.sett.s1.solver_library = "diffrax"
        inp.systems.sett.s1.solver_function = "newton_raphson"
        inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                                   atol=1e-6,
                                                   max_steps=50,
                                                   norm=jnp.linalg.norm,
                                                   kappa=0.01)
        # inp.systems.sett.s1.solver_library = "scipy"
        # inp.systems.sett.s1.solver_function = "root"
        # inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
        #                                            tolerance=1e-9)
        inp.systems.sett.s1.label = 'dq_001001'
        inp.systems.sett.s1.xloads.follower_forces = True
        inp.systems.sett.s1.xloads.follower_points = [[20, 1]]
        inp.systems.sett.s1.xloads.x = [0, 1, 2, 3]
        inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                              -50.,
                                                              -100.,
                                                              -300.
                                                              ]
                                                             ]

        inp.systems.sett.s1.t = [1, 2, 3]
        config = configuration.Config(inp)
        obj_sol = fem4inas.fem4inas_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/ArgyrisFrame"
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
