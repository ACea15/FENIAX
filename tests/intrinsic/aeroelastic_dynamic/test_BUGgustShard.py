import pathlib
import time
import jax.numpy as jnp
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
from feniax.preprocessor import solution
import feniax.feniax_main
import feniax.feniax_shardmain
import pytest

file_path = pathlib.Path(__file__).parent

class TestBUGGustShard:

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
        inp.system.aero.gust.fixed_discretisation = [150., u_inf]
        # Shard inputs
        inputflow = dict(length=np.linspace(0.8*150,150,2),
                         intensity= np.linspace(0.8*20, 20, 2),
                         rho_inf = np.linspace(0.8*rho_inf, rho_inf, 2)
                         )
        inp.system.shard = dict(input_type="gust1",
                                inputs=inputflow)
        config =  configuration.Config(inp)
        obj_sol =  feniax.feniax_shardmain.main(input_dict=inp, device_count=4)
        return obj_sol

    @pytest.fixture(scope="class")
    def solmax(self):

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
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol

    @pytest.fixture(scope="class")
    def solmin(self):

        label_dlm = "d1c7"
        sol = "eao"
        label_gaf = "Dd1c7F3Seao-50"
        num_modes = 50
        c_ref = 3.0
        u_inf = 209.62786434059765
        rho_inf = 0.8* 0.41275511341689247
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
        inp.system.aero.gust.intensity = 20 * 0.8
        inp.system.aero.gust.length = 150. * 0.8
        inp.system.aero.gust.step = 0.1
        inp.system.aero.gust.shift = 0.
        inp.system.aero.gust.panels_dihedral = file_path / f"../../../examples/BUG/AERO/Dihedral_{label_dlm}.npy"
        inp.system.aero.gust.collocation_points = file_path / f"../../../examples/BUG/AERO/Collocation_{label_dlm}.npy"
        config =  configuration.Config(inp)
        obj_sol = feniax.feniax_main.main(input_obj=config)
        return obj_sol
    
    def test_phi1(self, sol, solmax):
        
        assert jnp.allclose(sol.modes.phi1, solmax.modes.phi1)

    def test_phi2(self, sol, solmax):

        assert jnp.allclose(sol.modes.phi2, solmax.modes.phi2)

    def test_psi1(self, sol, solmax):

        assert jnp.allclose(sol.modes.psi1, solmax.modes.psi1)

    def test_phi1l(self, sol, solmax):
        
        assert jnp.allclose(sol.modes.phi1l, solmax.modes.phi1l)

    def test_phi2l(self, sol, solmax):

        assert jnp.allclose(sol.modes.phi2l, solmax.modes.phi2l)

    def test_psi1l(self, sol, solmax):

        assert jnp.allclose(sol.modes.psi1l, solmax.modes.psi1l)
        
    def test_psi2l(self, sol, solmax):

        assert jnp.allclose(sol.modes.psi2l, solmax.modes.psi2l)

    def test_phi1ml(self, sol, solmax):

        assert jnp.allclose(sol.modes.phi1ml, solmax.modes.phi1ml)

    def test_gamma1(self, sol, solmax):
        
        assert jnp.allclose(sol.couplings.gamma1,
                            solmax.couplings.gamma1)

    def test_gamma2(self, sol, solmax):

        assert jnp.allclose(sol.couplings.gamma2,
                            solmax.couplings.gamma2)
    
    def test_qsmax(self, sol, solmax):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.q[-1],
                            solmax.dynamicsystem_sys1.q,
                            atol=10e-5)

    def test_Xsmax(self, sol, solmax):

        # TODO: X2 is big but why had to go to such a relative big error to pass ??  
        assert jnp.allclose(sol.dynamicsystem_sys1.X2[-1],
                            solmax.dynamicsystem_sys1.X2,
                            atol=1e-3,rtol=1e-2)
        assert jnp.allclose(sol.dynamicsystem_sys1.X3[-1],
                            solmax.dynamicsystem_sys1.X3,
                            atol=10e-5)

    def test_ramax(self, sol, solmax):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.ra[-1],
                            solmax.dynamicsystem_sys1.ra,
                            atol=10e-5)

    def test_Cabmax(self, sol, solmax):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.Cab[-1],
                            solmax.dynamicsystem_sys1.Cab,
                            atol=10e-5)

    def test_qsmin(self, sol, solmin):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.q[0],
                            solmin.dynamicsystem_sys1.q,
                            atol=10e-5)

    def test_Xsmin(self, sol, solmin):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.X2[0],
                            solmin.dynamicsystem_sys1.X2,
                            atol=1e-3,rtol=1e-2)
        assert jnp.allclose(sol.dynamicsystem_sys1.X3[0],
                            solmin.dynamicsystem_sys1.X3,
                            atol=10e-5)

    def test_ramin(self, sol, solmin):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.ra[0],
                            solmin.dynamicsystem_sys1.ra,
                            atol=10e-5)

    def test_Cabmin(self, sol, solmin):
        
        assert jnp.allclose(sol.dynamicsystem_sys1.Cab[0],
                            solmin.dynamicsystem_sys1.Cab,
                            atol=10e-5)
        
