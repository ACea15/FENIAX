import pathlib
import time
import jax.numpy as jnp
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
from feniax.preprocessor import solution
import feniax.feniax_main
import pytest
import feniax.intrinsic.objectives as objectives

file_path = pathlib.Path(__file__).parent

class TestBUGGustclampedAD:

    epsilon_l = 1e-4
    epsilon_i = 1e-2
    epsilon_r = 1e-5
    
    def inputs1(self):

        label_dlm = "d1c7"
        sol = "cao"
        label_gaf = "Dd1c7F3Scao-50"
        num_modes = 50
        c_ref = 3.0
        u_inf = 209.62786434059765
        rho_inf = 0.41275511341689247
        num_poles = 5
        Dhj_file = f"D{label_gaf}p{num_poles}"
        Ahh_file = f"A{label_gaf}p{num_poles}"
        Poles_file = f"Poles{label_gaf}p{num_poles}"
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.eig_type = "inputs"

        inp.fem.connectivity = dict(# FusWing=['RWing',
                                    #          'LWing'],
                                    FusBack=['FusTail',
                                             'VTP'],
                                    FusFront=None,
                                    RWing=None,
                                    LWing=None,
                                    FusTail=None,
                                    VTP=['HTP', 'VTPTail'],
                                    HTP=['RHTP', 'LHTP'],
                                    VTPTail=None,
                                    RHTP=None,
                                    LHTP=None,
                                    )
        inp.fem.grid = file_path / f"../../../examples/BUG/FEM/structuralGrid_{sol[:-1]}"
        inp.fem.Ka_name = file_path / f"../../../examples/BUG/FEM/Ka_{sol[:-1]}.npy"
        inp.fem.Ma_name = file_path / f"../../../examples/BUG/FEM/Ma_{sol[:-1]}.npy"
        inp.fem.eig_names = [file_path / f"../../../examples/BUG/FEM/eigenvals_{sol}{num_modes}.npy",
                             file_path / f"../../../examples/BUG/FEM/eigenvecs_{sol}{num_modes}.npy"]
        inp.driver.typeof = "intrinsic"
        inp.driver.save_fem = False
        inp.driver.sol_path = None
        
        inp.fem.num_modes = num_modes

        inp.simulation.typeof = "single"
        inp.system.save = False
        inp.system.solution = "dynamic"
        inp.system.t1 = 1.
        inp.system.tn = 1001
        inp.system.solver_library = "runge_kutta"
        inp.system.solver_function = "ode"
        inp.system.solver_settings = dict(solver_name="rk4")
        inp.system.xloads.modalaero_forces = True
        inp.system.aero.c_ref = c_ref
        inp.system.aero.u_inf = u_inf
        inp.system.aero.rho_inf = rho_inf
        inp.system.aero.poles = file_path / f"../../../examples/BUG/AERO/{Poles_file}.npy"
        inp.system.aero.A = file_path / f"../../../examples/BUG/AERO/{Ahh_file}.npy"
        inp.system.aero.D = file_path / f"../../../examples/BUG/AERO/{Dhj_file}.npy"
        inp.system.aero.gust_profile = "mc"
        inp.system.aero.gust.intensity = 20.
        inp.system.aero.gust.length = 150.
        inp.system.aero.gust.step = 0.1
        inp.system.aero.gust.shift = 0.
        inp.system.aero.gust.panels_dihedral = file_path / f"../../../examples/BUG/AERO/Dihedral_{label_dlm}.npy"
        inp.system.aero.gust.collocation_points = file_path / f"../../../examples/BUG/AERO/Collocation_{label_dlm}.npy"

        return inp

    @pytest.fixture(scope="class")
    def sol_ad(self):
        inp = self.inputs1()
        inp.system.ad = dict(inputs=dict(length = 150.,
                                         intensity = 20.,
                                         u_inf=209.62786434059765,
                                         rho_inf=0.41275511341689247),
                             input_type="gust1",
                             grad_type="jacrev",
                             objective_fun="max",
                             objective_var="X2",
                             objective_args=dict(nodes=(11,), components=(0,1,2,3,4,5))
                             )
        
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture(scope="class")
    def sol_0(self):
        inp = self.inputs1()
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([11]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj
    
    @pytest.fixture(scope="class")
    def sol_length(self):
        inp = self.inputs1()
        inp.system.aero.gust.length = 150. + self.epsilon_l
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([11]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj

    @pytest.fixture(scope="class")
    def sol_intensity(self):
        inp = self.inputs1()
        inp.system.aero.gust.intensity = 20. + self.epsilon_i
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([11]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj

    @pytest.fixture(scope="class")
    def sol_rho(self):
        inp = self.inputs1()
        inp.system.aero.rho_inf = 0.41275511341689247 + self.epsilon_r
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([11]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj
    
    def test_lengthAD(self, sol_ad, sol_0, sol_length):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['length']
        derv_fd = (sol_length - sol_0) / self.epsilon_l
        assert jnp.linalg.norm(deriv_ad - derv_fd)/jnp.linalg.norm(deriv_ad) < 5e-6

    def test_intensityAD(self, sol_ad, sol_0, sol_intensity):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['intensity']
        derv_fd = (sol_intensity - sol_0) / self.epsilon_i
        assert jnp.linalg.norm(deriv_ad - derv_fd)/jnp.linalg.norm(deriv_ad) < 5e-5

    def test_rhoAD(self, sol_ad, sol_0, sol_rho):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['rho_inf']
        derv_fd = (sol_rho - sol_0) / self.epsilon_r
        assert jnp.linalg.norm(deriv_ad - derv_fd)/jnp.linalg.norm(deriv_ad) < 5e-5
        
    
class TestBUGGustfreeAD:

    epsilon_l = 1e-4
    epsilon_i = 1e-2
    epsilon_r = 1e-5
    
    def inputs1(self):

        label_dlm = "d1c7"
        sol = "eao"
        label_gaf = "Dd1c7F3Seao-50"
        num_modes = 50
        c_ref = 3.0
        u_inf = 209.62786434059765
        rho_inf = 0.41275511341689247
        num_poles = 5
        Dhj_file = f"D{label_gaf}p{num_poles}"
        Ahh_file = f"A{label_gaf}p{num_poles}"
        Poles_file = f"Poles{label_gaf}p{num_poles}"
        inp = Inputs()
        inp.engine = "intrinsicmodal"
        inp.fem.eig_type = "inputs"

        inp.fem.connectivity = dict(# FusWing=['RWing',
                                    #          'LWing'],
                                    FusBack=['FusTail',
                                             'VTP'],
                                    FusFront=None,
                                    RWing=None,
                                    LWing=None,
                                    FusTail=None,
                                    VTP=['HTP', 'VTPTail'],
                                    HTP=['RHTP', 'LHTP'],
                                    VTPTail=None,
                                    RHTP=None,
                                    LHTP=None,
                                    )
        inp.fem.grid = file_path / f"../../../examples/BUG/FEM/structuralGrid_{sol[:-1]}"
        inp.fem.Ka_name = file_path / f"../../../examples/BUG/FEM/Ka_{sol[:-1]}.npy"
        inp.fem.Ma_name = file_path / f"../../../examples/BUG/FEM/Ma_{sol[:-1]}.npy"
        inp.fem.eig_names = [file_path / f"../../../examples/BUG/FEM/eigenvals_{sol}{num_modes}.npy",
                             file_path / f"../../../examples/BUG/FEM/eigenvecs_{sol}{num_modes}.npy"]
        inp.driver.typeof = "intrinsic"
        inp.driver.save_fem = False
        inp.driver.sol_path = None
        
        inp.fem.num_modes = num_modes

        inp.simulation.typeof = "single"
        inp.system.save = False
        inp.system.solution = "dynamic"
        inp.system.bc1 = 'free'
        inp.system.q0treatment = 1  
        inp.system.t1 = 1.
        inp.system.tn = 1001
        inp.system.solver_library = "runge_kutta"
        inp.system.solver_function = "ode"
        inp.system.solver_settings = dict(solver_name="rk4")
        inp.system.xloads.modalaero_forces = True
        inp.system.aero.c_ref = c_ref
        inp.system.aero.u_inf = u_inf
        inp.system.aero.rho_inf = rho_inf
        inp.system.aero.poles = file_path / f"../../../examples/BUG/AERO/{Poles_file}.npy"
        inp.system.aero.A = file_path / f"../../../examples/BUG/AERO/{Ahh_file}.npy"
        inp.system.aero.D = file_path / f"../../../examples/BUG/AERO/{Dhj_file}.npy"
        inp.system.aero.gust_profile = "mc"
        inp.system.aero.gust.intensity = 20
        inp.system.aero.gust.length = 150.
        inp.system.aero.gust.step = 0.1
        inp.system.aero.gust.shift = 0.
        inp.system.aero.gust.panels_dihedral = file_path / f"../../../examples/BUG/AERO/Dihedral_{label_dlm}.npy"
        inp.system.aero.gust.collocation_points = file_path / f"../../../examples/BUG/AERO/Collocation_{label_dlm}.npy"

        return inp

    @pytest.fixture(scope="class")
    def sol_ad(self):
        inp = self.inputs1()
        inp.system.ad = dict(inputs=dict(length = 150.,
                                         intensity = 20.,
                                         u_inf=209.62786434059765,
                                         rho_inf=0.41275511341689247),
                             input_type="gust1",
                             grad_type="jacrev",
                             objective_fun="max",
                             objective_var="X2",
                             objective_args=dict(nodes=(11,), components=(0,1,2,3,4,5))
                             )
        
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture(scope="class")
    def sol_0(self):
        inp = self.inputs1()
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([11]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj
    
    @pytest.fixture(scope="class")
    def sol_length(self):
        inp = self.inputs1()
        inp.system.aero.gust.length = 150. + self.epsilon_l
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([11]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj

    @pytest.fixture(scope="class")
    def sol_intensity(self):
        inp = self.inputs1()
        inp.system.aero.gust.intensity = 20. + self.epsilon_i
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([11]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj

    @pytest.fixture(scope="class")
    def sol_rho(self):
        inp = self.inputs1()
        inp.system.aero.rho_inf = 0.41275511341689247 + self.epsilon_r
        config = configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        fobj = objectives.X2_MAX(obj_sol.dynamicsystem_sys1.X2,
                                 nodes=jnp.array([11]),
                                 components=jnp.array([0,1,2,3,4,5]),
                                 t=jnp.arange(config.system.tn)
                                 )
        
        return fobj
    
    def test_lengthAD(self, sol_ad, sol_0, sol_length):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['length']
        derv_fd = (sol_length - sol_0) / self.epsilon_l
        assert jnp.linalg.norm(deriv_ad - derv_fd)/jnp.linalg.norm(deriv_ad) < 5e-6

    def test_intensityAD(self, sol_ad, sol_0, sol_intensity):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['intensity']
        derv_fd = (sol_intensity - sol_0) / self.epsilon_i
        assert jnp.linalg.norm(deriv_ad - derv_fd)/jnp.linalg.norm(deriv_ad) < 5e-5

    def test_rhoAD(self, sol_ad, sol_0, sol_rho):

        deriv_ad = sol_ad.dynamicsystem_sys1.jac['rho_inf']
        derv_fd = (sol_rho - sol_0) / self.epsilon_r
        assert jnp.linalg.norm(deriv_ad - derv_fd)/jnp.linalg.norm(deriv_ad) < 5e-5
    
