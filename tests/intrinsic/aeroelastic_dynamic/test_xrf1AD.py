import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor import solution
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import pytest
import pathlib
import feniax.intrinsic.objectives as objectives

file_path = pathlib.Path(__file__).parent

@pytest.mark.private
class TestXRF1:
    
    epsilon = 1e-4
    #@pytest.fixture(scope="class")
    def inputs1(self):
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
        inp.systems.sett.s1.t1 = 1.
        inp.systems.sett.s1.tn = 1001
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
        
        return inputs

    @pytest.fixture
    def sol_ad(self):
        inp = self.inputs1()
        inp.system.ad = dict(inputs=dict(length = 67.,
                                         intensity = 14.0732311562*2,
                                         u_inf=200.,
                                         rho_inf = 1.225),
                             input_type="gust1",
                             grad_type="jacrev",
                             objective_fun="max",
                             objective_var="X2",
                             objective_args=dict(nodes=(5,), components=(0,1,2,3,4,5))
                             )
        
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol
    
    @pytest.fixture
    def sol_0(self):
        inp = self.inputs1()
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([5]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj
    
    @pytest.fixture
    def sol_length(self):
        inp = self.inputs1()
        inp.system.aero.gust.length = 67. + self.epsilon
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([5]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj

    @pytest.fixture
    def sol_intensity(self):
        inp = self.inputs1()
        inp.system.aero.gust.intensity = 14.0732311562*2 + self.epsilon
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([5]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj

    @pytest.fixture
    def sol_rho(self):
        inp = self.inputs1()
        inp.system.aero.rho_inf = 1.225 + self.epsilon
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([5]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj
    
    def test_lengthAD(self, sol_ad, sol_0, sol_length):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['length']
        derv_fd = (sol_length - sol_0) / self.epsilon
        assert jnp.linalg.norm(deriv_ad - derv_fd) < 1e-2

    def test_intensityAD(self, sol_ad, sol_0, sol_intensity):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['intensity']
        derv_fd = (sol_intensity - sol_0) / self.epsilon
        assert jnp.linalg.norm(deriv_ad - derv_fd) < 1e-2

    def test_rhoAD(self, sol_ad, sol_0, sol_rho):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['rho']
        derv_fd = (sol_rho - sol_0) / self.epsilon
        assert jnp.linalg.norm(deriv_ad - derv_fd) < 1e-2
        
