import numpy as np
import pathlib
import pytest

from feniax.preprocessor.inputs import Inputs
import feniax.feniax_shardmain


file_path = pathlib.Path(__file__).parent

@pytest.mark.slow
class TestBUGForager:

    c_ref = 3.0
    u_inf = 209.62786434059765
    rho_inf = 0.41275511341689247
    epsilon  =1e-5
    node = 13
    components = (2,3,4) # shear, torsion, oop bending
    tn = 2501
    
    def get_inputs(self):
        
        label_dlm = "d1c7"
        sol = "eao"
        label_gaf = "Dd1c7F3Seao-100"
        num_modes = 100
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
        inp.fem.num_modes = num_modes

        inp.simulation.typeof = "single"
        inp.system.name = "s1"
        if sol[0] == "e": # free model, otherwise clamped
            inp.system.bc1 = 'free'
            inp.system.q0treatment = 1
        inp.system.solution = "dynamic"
        inp.system.t1 = 1.
        inp.system.tn = self.tn
        inp.system.solver_library = "runge_kutta"
        inp.system.solver_function = "ode"
        inp.system.solver_settings = dict(solver_name="rk4")
        inp.system.xloads.modalaero_forces = True
        inp.system.aero.c_ref = self.c_ref
        inp.system.aero.u_inf = self.u_inf
        inp.system.aero.rho_inf = self.rho_inf
        inp.system.aero.poles = file_path / f"../../../examples/BUG/AERO/{Poles_file}.npy"
        inp.system.aero.A = file_path / f"../../../examples/BUG/AERO/{Ahh_file}.npy"
        inp.system.aero.D = file_path / f"../../../examples/BUG/AERO/{Dhj_file}.npy"
        inp.system.aero.gust_profile = "mc"
        inp.system.aero.gust.intensity = 30 #25
        inp.system.aero.gust.length = 200. #150.
        inp.system.aero.gust.step = 0.1
        inp.system.aero.gust.shift = 0.
        inp.system.aero.gust.panels_dihedral = file_path / f"../../../examples/BUG/AERO/Dihedral_{label_dlm}.npy"
        inp.system.aero.gust.collocation_points = file_path / f"../../../examples/BUG/AERO/Collocation_{label_dlm}.npy"
        inp.system.save = False        
        inp.driver.save_fem = False
        inp.driver.sol_path = None
        inp.system.aero.gust.fixed_discretisation = [150, self.u_inf]
        return inp

    @pytest.fixture(scope="class")
    def solforager(self):
    
        # Shard inputs
        inp = self.get_inputs()
        inputflow = dict(length=np.linspace(50,200,16),
                         intensity=np.linspace(5, 25, 2),
                         rho_inf = [0.8*self.rho_inf, self.rho_inf] #np.linspace(0.34,0.48,1)
                         )

        inp.system.shard = dict(input_type="gust1",
                                inputs=inputflow)
        inp.system.operationalmode = "shardmap"   
        node = 13
        components = (2,3,4) # shear, torsion, oop bending
        inp.forager.typeof = "shard2adgust"
        inp.forager.settings.gathersystem_name = "s1"
        inp.forager.settings.scattersystems_name = "scatter"
        inp.forager.settings.ad = dict(inputs=dict(length = None,
                                                   intensity = None,
                                                   u_inf=self.u_inf,
                                                   rho_inf = None),
                                       input_type="gust1",
                                       grad_type="jacrev",
                                       objective_fun="max",
                                       objective_var="X2",
                                       objective_args=dict(nodes=(node,),
                                                           components=components)
                                       )

        num_gpus = 1
        solforager = feniax.feniax_shardmain.main(input_dict=inp, device_count=num_gpus)

        return solforager

    @pytest.fixture(scope="class")
    def sol(self, solforager):
    
        # Shard inputs
        inp = self.get_inputs()
        import feniax.feniax_main
        import jax.numpy as jnp
        import feniax.intrinsic.objectives as objectives

        inp.system.operationalmode = ""
        inp.system.shard = None
        inp.forager = None
        t_range = jnp.arange(inp.system.tn)
        index_i = list(solforager.forager_shard2adgust.filtered_indexes)[0]
        rho, uinf, length, intensity = solforager.shards_s1.points[index_i]
        inp.system.aero.rho_inf = rho 
        inp.system.aero.u_inf = uinf
        inp.system.aero.gust.length = length
        inp.system.aero.gust.intensity = intensity
        sol = feniax.feniax_main.main(input_dict=inp)

        return sol

    @pytest.fixture(scope="class")
    def sol_rho(self, solforager):
    
        # Shard inputs
        inp = self.get_inputs()
        import feniax.feniax_main

        inp.system.operationalmode = ""
        inp.system.shard = None
        inp.forager = None
        index_i = list(solforager.forager_shard2adgust.filtered_indexes)[0]
        rho, uinf, length, intensity = solforager.shards_s1.points[index_i]
        inp.system.aero.rho_inf = rho 
        inp.system.aero.u_inf = uinf
        inp.system.aero.gust.length = length
        inp.system.aero.gust.intensity = intensity
        inp.system.aero.rho_inf += self.epsilon
        solrho = feniax.feniax_main.main(input_dict=inp)
        return solrho
    
    @pytest.fixture(scope="class")
    def sol_length(self, solforager):
    
        # Shard inputs
        inp = self.get_inputs()
        import feniax.feniax_main

        inp.system.operationalmode = ""
        inp.system.shard = None
        inp.forager = None
        index_i = list(solforager.forager_shard2adgust.filtered_indexes)[0]
        rho, uinf, length, intensity = solforager.shards_s1.points[index_i]
        inp.system.aero.rho_inf = rho 
        inp.system.aero.u_inf = uinf
        inp.system.aero.gust.length = length
        inp.system.aero.gust.intensity = intensity
        inp.system.aero.gust.length += self.epsilon
        sollength = feniax.feniax_main.main(input_dict=inp)
        return sollength

    @pytest.fixture(scope="class")
    def sol_intensity(self, solforager):
    
        # Shard inputs
        inp = self.get_inputs()
        import feniax.feniax_main

        inp.system.operationalmode = ""
        inp.system.shard = None
        inp.forager = None
        index_i = list(solforager.forager_shard2adgust.filtered_indexes)[0]
        rho, uinf, length, intensity = solforager.shards_s1.points[index_i]
        inp.system.aero.rho_inf = rho 
        inp.system.aero.u_inf = uinf
        inp.system.aero.gust.length = length
        inp.system.aero.gust.intensity = intensity
        inp.system.aero.gust.intensity += self.epsilon
        solintensity = feniax.feniax_main.main(input_dict=inp)
        return solintensity    
    
    def test_lengthAD(self, sol, sol_length, solforager):
        
        import jax.numpy as jnp
        import feniax.intrinsic.objectives as objectives

        t_range = jnp.arange(self.tn)
        jac_fd = (objectives.X2_MAX(sol_length.dynamicsystem_s1.X2,
                                     jnp.array([self.node]),
                                     jnp.array(self.components),
                                     t_range) -
                      objectives.X2_MAX(sol.dynamicsystem_s1.X2,
                                        jnp.array([self.node]),
                                        jnp.array(self.components),
                                        t_range)
                      ) / self.epsilon
        
        jac_ad = solforager.dynamicsystem_scatter0.jac['length']
        assert jnp.linalg.norm(jac_ad - jac_fd)/jnp.linalg.norm(jac_ad) < 1e-5

    def test_intensityAD(self, sol, sol_intensity, solforager):

        import jax.numpy as jnp
        import feniax.intrinsic.objectives as objectives

        t_range = jnp.arange(self.tn)
        jac_fd = (objectives.X2_MAX(sol_intensity.dynamicsystem_s1.X2,
                                     jnp.array([self.node]),
                                     jnp.array(self.components),
                                     t_range) -
                      objectives.X2_MAX(sol.dynamicsystem_s1.X2,
                                        jnp.array([self.node]),
                                        jnp.array(self.components),
                                        t_range)
                      ) / self.epsilon
        
        jac_ad = solforager.dynamicsystem_scatter0.jac['intensity']
        assert jnp.linalg.norm(jac_ad - jac_fd)/jnp.linalg.norm(jac_ad) < 1e-5

    def test_rhoAD(self, sol, sol_rho, solforager):

        import jax.numpy as jnp
        import feniax.intrinsic.objectives as objectives

        t_range = jnp.arange(self.tn) 
        jac_fd = (objectives.X2_MAX(sol_rho.dynamicsystem_s1.X2,
                                     jnp.array([self.node]),
                                     jnp.array(self.components),
                                     t_range) -
                      objectives.X2_MAX(sol.dynamicsystem_s1.X2,
                                        jnp.array([self.node]),
                                        jnp.array(self.components),
                                        t_range)
                      ) / self.epsilon
        
        jac_ad = solforager.dynamicsystem_scatter0.jac['rho_inf']
        assert jnp.linalg.norm(jac_ad - jac_fd)/jnp.linalg.norm(jac_ad) < 5e-5

    def test_max(self, solforager):

        import jax.numpy as jnp
        # for component 2:
        ci = 2
        field_i = jnp.abs(solforager.dynamicsystem_s1.X2[:,:,ci,self.node])
        field_ivalue = jnp.max(field_i)
        argmax = jnp.argmax(field_i)
        index = jnp.unravel_index(argmax,
                                  field_i.shape) # get max index in field_i shape
        assert solforager.forager_shard2adgust.filtered_map[(self.node,ci)] == index
        # for component 3:
        ci = 3
        field_i = jnp.abs(solforager.dynamicsystem_s1.X2[:,:,ci,self.node])
        field_ivalue = jnp.max(field_i)
        argmax = jnp.argmax(field_i)
        index = jnp.unravel_index(argmax,
                                  field_i.shape) # get max index in field_i shape
        assert solforager.forager_shard2adgust.filtered_map[(self.node,ci)] == index

        # for component 4:
        ci = 4
        field_i = jnp.abs(solforager.dynamicsystem_s1.X2[:,:,ci,self.node])
        field_ivalue = jnp.max(field_i)
        argmax = jnp.argmax(field_i)
        index = jnp.unravel_index(argmax,
                                  field_i.shape) # get max index in field_i shape
        assert solforager.forager_shard2adgust.filtered_map[(self.node,ci)] == index
