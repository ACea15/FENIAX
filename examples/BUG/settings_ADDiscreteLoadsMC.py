# [[file:modelgeneration.org::*High loading][High loading:1]]
import pathlib
import time
#import jax.numpy as jnp
import numpy as np
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_shardmain
sol = "cao"
num_modes = 100
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
inp.driver.typeof = "intrinsic"
inp.driver.sol_path = pathlib.Path(
    f"./results/DiscreteMC1high{num_modes}")

inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "static"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "newton"
inp.system.solver_settings = dict(rtol=1e-6,
                                  atol=1e-6,
                                  max_steps=50,
                                  norm="linalg_norm",
                                  kappa=0.01)
inp.system.xloads.follower_forces = True
inp.system.xloads.x = [0, 1, 2, 3, 4, 5]
inp.system.t = [1, 2, 3, 4, 5]
# rwing: 14-35
# lwing: 40-61
points = []
interpolation = []
_interpolation = [0., 3.e3, 7e3, 9e3, 1e4, 1.5e4] # 1.5e4, 2e4..4e4] #[0., 0., 0., 0.]
_interpolation_torsion = [i*2 for i in _interpolation] #[0., 4e3, 1e4, 2e4, 4e4, 5e4] 
for ri,li in zip(range(14, 36),range(40,62)):
    points.append([ri, 2])
    points.append([ri, 4])
    points.append([li, 2])
    points.append([li, 4])
for i, _ in enumerate(range(len(points))):

    if i % 2 == 0:
        interpolation.append(_interpolation)
    else:
        interpolation.append(_interpolation_torsion)

interpolation = np.array(interpolation)  # num_pointforces x num_interpolation  
paths = 8*4 #200
sigma0 = 0.15  # percentage of mu for sigma
mu = _interpolation[-1]
sigma = (sigma0) * _interpolation[-1]
rand = np.random.normal(mu, sigma, paths)
mu_torsion = _interpolation_torsion[-1]
sigma_torsion = (sigma0) * _interpolation_torsion[-1]
rand_torsion = np.random.normal(mu_torsion, sigma_torsion, paths)
follower_interpolation = []
for i, ri in enumerate(rand):
    interpolationrand = np.copy(interpolation)
    interpolationrand[::2, -1] = ri
    interpolationrand[1::2, -1] = rand_torsion[i]
    follower_interpolation.append(interpolationrand)
#follower_interpolation = [interpolation * ri for ri in rand]
follower_points = [points]*paths
inputforces = dict(follower_points=follower_points,
                   follower_interpolation=follower_interpolation
                   )
#inp.system.operationalmode = "shardmap"
inp.system.shard = dict(input_type="pointforces",
                        inputs=inputforces)
inp.system.ad = dict(inputs=dict(eigenvals=inp.fem.eig_names[0],
                                 eigenvecs=inp.fem.eig_names[1],
                                 Ka=inp.fem.Ka_name,
                                 Ma=inp.fem.Ma_name),
                     input_type="fem",
                     grad_type="jacrev", #"jacrev", #value
                     objective_fun="pmean",
                     objective_var="ra",
                     objective_args=dict(nodes=(13,), components=(0,1),
                                         t=(inp.system.t[-1],))
                     )

sol1 = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)

# np.mean(sol.staticsystem_sys1.ra[:,-1,2,35])
# np.std(sol.staticsystem_sys1.ra[:,-1,2,35])
# High loading:1 ends here

# [[file:modelgeneration.org::*Small loading][Small loading:1]]
# import pathlib
# import time
# #import jax.numpy as jnp
# import numpy as np
# from feniax.preprocessor.inputs import Inputs
# import feniax.feniax_shardmain
# sol = "cao"
# num_modes = 50
# inp = Inputs()
# inp.engine = "intrinsicmodal"
# inp.fem.eig_type = "inputs"

# inp.fem.connectivity = dict(# FusWing=['RWing',
#                             #          'LWing'],
#                             FusBack=['FusTail',
#                                      'VTP'],
#                             FusFront=None,
#                             RWing=None,
#                             LWing=None,
#                             FusTail=None,
#                             VTP=['HTP', 'VTPTail'],
#                             HTP=['RHTP', 'LHTP'],
#                             VTPTail=None,
#                             RHTP=None,
#                             LHTP=None,
#                             )
# inp.fem.grid = f"./FEM/structuralGrid_{sol[:-1]}"
# #inp.fem.folder = pathlib.Path('./FEM/')
# inp.fem.Ka_name = f"./FEM/Ka_{sol[:-1]}.npy"
# inp.fem.Ma_name = f"./FEM/Ma_{sol[:-1]}.npy"
# inp.fem.eig_names = [f"./FEM/eigenvals_{sol}{num_modes}.npy",
#                      f"./FEM/eigenvecs_{sol}{num_modes}.npy"]
# inp.driver.typeof = "intrinsic"
# inp.fem.num_modes = num_modes
# inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     "./results/DiscreteMC1small")

# inp.simulation.typeof = "single"
# inp.system.name = "s1"
# inp.system.solution = "static"
# inp.system.solver_library = "diffrax"
# inp.system.solver_function = "newton"
# inp.system.solver_settings = dict(rtol=1e-6,
#                                   atol=1e-6,
#                                   max_steps=50,
#                                   norm="linalg_norm",
#                                   kappa=0.01)
# inp.system.xloads.follower_forces = True
# inp.system.xloads.x = [0, 1, 2, 3, 4, 5]
# inp.system.t = [1, 2, 3, 4, 5]
# # rwing: 14-35
# # lwing: 40-61
# points = []
# interpolation = []
# _interpolationsmall = [i*1e-2 for i in _interpolation] 
# _interpolationsmall_torsion = [i*1e-2 for i in _interpolation_torsion]
# for ri,li in zip(range(14, 36),range(40,62)):
#     points.append([ri, 2])
#     points.append([ri, 4])
#     points.append([li, 2])
#     points.append([li, 4])
# for i, _ in enumerate(range(len(points))):

#     if i % 2 == 0:
#         interpolation.append(_interpolationsmall)
#     else:
#         interpolation.append(_interpolationsmall_torsion)

# interpolation = np.array(interpolation)  # num_pointforces x num_interpolation  
# paths = 8*10
# sigma0 = 0.15  # percentage of mu for sigma
# mu = _interpolationsmall[-1]
# sigma = (sigma0) * _interpolationsmall[-1]
# rand = np.random.normal(mu, sigma, paths)
# mu_torsion = _interpolationsmall_torsion[-1]
# sigma_torsion = (sigma0) * _interpolationsmall_torsion[-1]
# rand_torsion = np.random.normal(mu_torsion, sigma_torsion, paths)
# follower_interpolation = []
# for i, ri in enumerate(rand):
#     interpolationrand = np.copy(interpolation)
#     interpolationrand[::2, -1] = ri
#     interpolationrand[1::2, -1] = rand_torsion[i]
#     follower_interpolation.append(interpolationrand)
# #follower_interpolation = [interpolation * ri for ri in rand]
# follower_points = [points]*paths
# inputforces = dict(follower_points=follower_points,
#                    follower_interpolation=follower_interpolation
#                    )
# inp.system.shard = dict(input_type="pointforces",
#                         inputs=inputforces)

# sol2 = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)
# # np.mean(sol.staticsystem_sys1.ra[:,-1,2,35])
# # np.std(sol.staticsystem_sys1.ra[:,-1,2,35])
# # Small loading:1 ends here

# # [[file:modelgeneration.org::*Very Small loading][Very Small loading:1]]
# import pathlib
# import time
# #import jax.numpy as jnp
# import numpy as np
# from feniax.preprocessor.inputs import Inputs
# import feniax.feniax_shardmain
# sol = "cao"
# num_modes = 50
# inp = Inputs()
# inp.engine = "intrinsicmodal"
# inp.fem.eig_type = "inputs"

# inp.fem.connectivity = dict(# FusWing=['RWing',
#                             #          'LWing'],
#                             FusBack=['FusTail',
#                                      'VTP'],
#                             FusFront=None,
#                             RWing=None,
#                             LWing=None,
#                             FusTail=None,
#                             VTP=['HTP', 'VTPTail'],
#                             HTP=['RHTP', 'LHTP'],
#                             VTPTail=None,
#                             RHTP=None,
#                             LHTP=None,
#                             )
# inp.fem.grid = f"./FEM/structuralGrid_{sol[:-1]}"
# #inp.fem.folder = pathlib.Path('./FEM/')
# inp.fem.Ka_name = f"./FEM/Ka_{sol[:-1]}.npy"
# inp.fem.Ma_name = f"./FEM/Ma_{sol[:-1]}.npy"
# inp.fem.eig_names = [f"./FEM/eigenvals_{sol}{num_modes}.npy",
#                      f"./FEM/eigenvecs_{sol}{num_modes}.npy"]
# inp.driver.typeof = "intrinsic"
# inp.fem.num_modes = num_modes
# inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     "./results/DiscreteMC1verysmall")

# inp.simulation.typeof = "single"
# inp.system.name = "s1"
# inp.system.solution = "static"
# inp.system.solver_library = "diffrax"
# inp.system.solver_function = "newton"
# inp.system.solver_settings = dict(rtol=1e-6,
#                                   atol=1e-6,
#                                   max_steps=50,
#                                   norm="linalg_norm",
#                                   kappa=0.01)
# inp.system.xloads.follower_forces = True
# inp.system.xloads.x = [0, 1, 2, 3, 4, 5]
# inp.system.t = [1, 2, 3, 4, 5]
# # rwing: 14-35
# # lwing: 40-61
# points = []
# interpolation = []
# _interpolationverysmall = [i*1e-3 for i in _interpolation] 
# _interpolationverysmall_torsion = [i*1e-3 for i in _interpolation_torsion]   

# for ri,li in zip(range(14, 36),range(40,62)):
#     points.append([ri, 2])
#     points.append([ri, 4])
#     points.append([li, 2])
#     points.append([li, 4])
# for i, _ in enumerate(range(len(points))):

#     if i % 2 == 0:
#         interpolation.append(_interpolationverysmall)
#     else:
#         interpolation.append(_interpolationverysmall_torsion)

# interpolation = np.array(interpolation)  # num_pointforces x num_interpolationverysmall  
# paths = 8*10
# sigma0 = 0.15  # percentage of mu for sigma
# mu = _interpolationverysmall[-1]
# sigma = (sigma0) * _interpolationverysmall[-1]
# rand = np.random.normal(mu, sigma, paths)
# mu_torsion = _interpolationverysmall_torsion[-1]
# sigma_torsion = (sigma0) * _interpolationverysmall_torsion[-1]
# rand_torsion = np.random.normal(mu_torsion, sigma_torsion, paths)
# follower_interpolation = []
# for i, ri in enumerate(rand):
#     interpolationrand = np.copy(interpolation)
#     interpolationrand[::2, -1] = ri
#     interpolationrand[1::2, -1] = rand_torsion[i]
#     follower_interpolation.append(interpolationrand)
# #follower_interpolation = [interpolation * ri for ri in rand]
# follower_points = [points]*paths
# inputforces = dict(follower_points=follower_points,
#                    follower_interpolation=follower_interpolation
#                    )
# inp.system.shard = dict(input_type="pointforces",
#                         inputs=inputforces)

# sol3 = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)
# # np.mean(sol.staticsystem_sys1.ra[:,-1,2,35])
# # np.std(sol.staticsystem_sys1.ra[:,-1,2,35])
# # Very Small loading:1 ends here

# # [[file:modelgeneration.org::*Statistics][Statistics:1]]
# u_mean = np.mean(sol1.staticsystem_sys1.ra[:,-1,2,35] - config.fem.X[35,2])
# u_std = np.std(sol1.staticsystem_sys1.ra[:,-1,2,35])

# print(f"Mean displacement node 35: {u_mean}")
# print(f"std displacement node 35: {u_std}") 
# print(f"Ratio displacement node 35: {u_mean/u_std}") 
# print("***************") 

# u_mean2 = np.mean(sol2.staticsystem_sys1.ra[:,-1,2,35] - config.fem.X[35,2])
# u_std2 = np.std(sol2.staticsystem_sys1.ra[:,-1,2,35])

# print(f"Mean displacement node 35: {u_mean2}")
# print(f"std displacement node 35: {u_std2}") 
# print(f"ratio displacement node 35: {u_mean2/u_std2}") 
# print("***************") 

# u_mean3 = np.mean(sol3.staticsystem_sys1.ra[:,-1,2,35] - config.fem.X[35,2])
# u_std3 = np.std(sol3.staticsystem_sys1.ra[:,-1,2,35])

# print(f"Mean displacement node 35: {u_mean3}")
# print(f"std displacement node 35: {u_std3}") 
# print(f"ratio displacement node 35: {u_mean3/u_std3}") 
# print("***************")
# # Statistics:1 ends here
