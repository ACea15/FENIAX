import pathlib
import time
import jax.numpy as jnp
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
from feniax.preprocessor import solution
import feniax.feniax_main
import pytest

file_path = pathlib.Path(__file__).parent

class TestBUGGustclamped:

    @pytest.fixture(scope="class")
    def sol(self):

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
        inp.system.operationalmode = "fast"        
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
        inp.system.aero.gust.intensity = 20
        inp.system.aero.gust.length = 150.
        inp.system.aero.gust.step = 0.1
        inp.system.aero.gust.shift = 0.
        inp.system.aero.gust.panels_dihedral = file_path / f"../../../examples/BUG/AERO/Dihedral_{label_dlm}.npy"
        inp.system.aero.gust.collocation_points = file_path / f"../../../examples/BUG/AERO/Collocation_{label_dlm}.npy"
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/BUG/gust2_cao"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("Modes")
        sol.load_container("Couplings")
        sol.load_container("DynamicSystem", label="_sys1")

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
        
        assert jnp.allclose(sol.dynamicsystem_sys1.q,
                            data.dynamicsystem_sys1.q)

    def test_Xs(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.X2,
                            data.dynamicsystem_sys1.X2,
                            atol=1e-5)
        assert jnp.allclose(sol.dynamicsystem_sys1.X3,
                            data.dynamicsystem_sys1.X3)

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.ra,
                            data.dynamicsystem_sys1.ra)

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.Cab,
                            data.dynamicsystem_sys1.Cab)

class TestBUGGustfree:

    @pytest.fixture(scope="class")
    def sol(self):

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
        inp.system.operationalmode = "fast"        
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
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture
    def data(self):
        sol_path = file_path / "data/BUG/gust2_eao"
        sol = solution.IntrinsicSolution(sol_path)
        sol.load_container("Modes")
        sol.load_container("Couplings")
        sol.load_container("DynamicSystem", label="_sys1")

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
        
        assert jnp.allclose(sol.dynamicsystem_sys1.q,
                            data.dynamicsystem_sys1.q)

    def test_Xs(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.X2,
                            data.dynamicsystem_sys1.X2,
                            atol=1e-5)
        assert jnp.allclose(sol.dynamicsystem_sys1.X3,
                            data.dynamicsystem_sys1.X3)

    def test_ra(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.ra,
                            data.dynamicsystem_sys1.ra)

    def test_Cab(self, sol, data):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.Cab,
                            data.dynamicsystem_sys1.Cab)
        
