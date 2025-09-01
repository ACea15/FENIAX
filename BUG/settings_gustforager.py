# [[file:modelgeneration.org::*Forager][Forager:1]]
import pathlib
import time
#import jax.numpy as jnp
import numpy as np
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_shardmain
import sys

if len(sys.argv) > 1:
    results_path = f"{sys.argv[1]}/results/"
else:
    results_path = "./results/"

label_dlm = "d1c7"
sol = "eao"
label_gaf = "Dd1c7F3Seao-100"
num_modes = 100
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
inp.fem.grid = f"./FEM/structuralGrid_{sol[:-1]}"
#inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.Ka_name = f"./FEM/Ka_{sol[:-1]}.npy"
inp.fem.Ma_name = f"./FEM/Ma_{sol[:-1]}.npy"
inp.fem.eig_names = [f"./FEM/eigenvals_{sol}{num_modes}.npy",
                     f"./FEM/eigenvecs_{sol}{num_modes}.npy"]
inp.driver.typeof = "intrinsic"
inp.fem.num_modes = num_modes

inp.simulation.typeof = "single"
inp.system.name = "s1"
if sol[0] == "e": # free model, otherwise clamped
    inp.system.bc1 = 'free'
    inp.system.q0treatment = 1
inp.system.solution = "dynamic"
inp.system.t1 = 1.
inp.system.tn = 2501
inp.system.solver_library = "runge_kutta"
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="rk4")
inp.system.xloads.modalaero_forces = True
inp.system.aero.c_ref = c_ref
inp.system.aero.u_inf = u_inf
inp.system.aero.rho_inf = rho_inf
inp.system.aero.poles = f"./AERO/{Poles_file}.npy"
inp.system.aero.A = f"./AERO/{Ahh_file}.npy"
inp.system.aero.D = f"./AERO/{Dhj_file}.npy"
inp.system.aero.gust_profile = "mc"
inp.system.aero.gust.intensity = 30 #25
inp.system.aero.gust.length = 200. #150.
inp.system.aero.gust.step = 0.1
inp.system.aero.gust.shift = 0.
inp.system.aero.gust.panels_dihedral = f"./AERO/Dihedral_{label_dlm}.npy"
inp.system.aero.gust.collocation_points = f"./AERO/Collocation_{label_dlm}.npy"

inp.driver.sol_path = pathlib.Path(
    f"{results_path}/gustforager")
inp.system.aero.gust.fixed_discretisation = [150, u_inf]
# Shard inputs
inputflow = dict(length=np.linspace(25,265,13),
                 #intensity=np.linspace(0.1, 3, 11),
                 rho_inf = np.linspace(0.34,0.48,8)
               )
inputflow = dict(length=np.linspace(50,200,16),
                 intensity=np.linspace(5, 25, 2),
                 rho_inf = [0.8*rho_inf, rho_inf] #np.linspace(0.34,0.48,1)
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
                                           u_inf=u_inf,
                                           rho_inf = None),
                               input_type="gust1",
                               grad_type="jacrev",
                               objective_fun="max",
                               objective_var="X2",
                               objective_args=dict(nodes=(node,),
                                                   components=components)
                               )

num_gpus = 8
solforager = feniax.feniax_shardmain.main(input_dict=inp, device_count=num_gpus)

def validation_max():

    import jax.numpy as jnp

    # for component 2:
    ci = 2
    field_i = jnp.abs(solforager.dynamicsystem_s1.X2[:,:,ci,node])
    field_ivalue = jnp.max(field_i)
    argmax = jnp.argmax(field_i)
    index = jnp.unravel_index(argmax,
                              field_i.shape) # get max index in field_i shape
    assert solforager.forager_shard2adgust.filtered_map[(node,ci)] == index
    # for component 3:
    ci = 3
    field_i = jnp.abs(solforager.dynamicsystem_s1.X2[:,:,ci,node])
    field_ivalue = jnp.max(field_i)
    argmax = jnp.argmax(field_i)
    index = jnp.unravel_index(argmax,
                              field_i.shape) # get max index in field_i shape
    assert solforager.forager_shard2adgust.filtered_map[(node,ci)] == index

    # for component 4:
    ci = 4
    field_i = jnp.abs(solforager.dynamicsystem_s1.X2[:,:,ci,node])
    field_ivalue = jnp.max(field_i)
    argmax = jnp.argmax(field_i)
    index = jnp.unravel_index(argmax,
                              field_i.shape) # get max index in field_i shape
    assert solforager.forager_shard2adgust.filtered_map[(node,ci)] == index

def validation_ad():
    """
    Only run the gust of the first problematic gust case to compare FD
    """
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
    inp.driver.sol_path = pathlib.Path(
        f"{results_path}/gustforager_validation")

    sol = feniax.feniax_main.main(input_dict=inp)
    epsilon = 1e-5
    inp.system.aero.rho_inf += epsilon
    inp.driver.sol_path = pathlib.Path(
        f"{results_path}/gustforager_epsilonrho")
    sol_rho = feniax.feniax_main.main(input_dict=inp)
    jac_rho = (objectives.X2_MAX(sol_rho.dynamicsystem_s1.X2,
                                 jnp.array([node]),
                                 jnp.array(components),
                                 t_range) -
               objectives.X2_MAX(sol.dynamicsystem_s1.X2,
                                 jnp.array([node]),
                                 jnp.array(components),
                                 t_range)
               ) / epsilon
    jnp.save(f"{results_path}/gustforager_epsilonrho/jac_rho.npy", jac_rho)      
    ##########
    epsilon = 1e-5 # tweaking fd
    inp.system.aero.rho_inf = rho
    inp.system.aero.gust.length += epsilon
    inp.driver.sol_path = pathlib.Path(
        f"{results_path}/gustforager_epsilonlength")
    sol_length = feniax.feniax_main.main(input_dict=inp)
    
    jac_length = (objectives.X2_MAX(sol_length.dynamicsystem_s1.X2,
                                 jnp.array([node]),
                                 jnp.array(components),
                                 t_range) -
                  objectives.X2_MAX(sol.dynamicsystem_s1.X2,
                                    jnp.array([node]),
                                    jnp.array(components),
                                    t_range)
                  ) / epsilon
    
    jnp.save(f"{results_path}/gustforager_epsilonlength/jac_length.npy", jac_length)
    ############
    epsilon = 1e-5 # tweaking fd      
    inp.system.aero.gust.length = length
    inp.system.aero.gust.intensity += epsilon
    inp.driver.sol_path = pathlib.Path(
        f"{results_path}/gustforager_epsilonintensity")
    sol_intensity = feniax.feniax_main.main(input_dict=inp)
    jac_intensity = (objectives.X2_MAX(sol_intensity.dynamicsystem_s1.X2,
                                       jnp.array([node]),
                                       jnp.array(components),
                                       t_range) -
                     objectives.X2_MAX(sol.dynamicsystem_s1.X2,
                                       jnp.array([node]),
                                       jnp.array(components),
                                       t_range)
               ) / epsilon
    jnp.save(f"{results_path}/gustforager_epsilonintensity/jac_intensity.npy", jac_intensity)
    return jac_rho, jac_length, jac_intensity

validation_max()
jac_rho, jac_length, jac_intensity = validation_ad()
# Forager:1 ends here
