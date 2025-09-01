# [[file:modelgeneration.org::*High loading][High loading:1]]
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
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/ShardingMC_AD11")

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
paths = 8*4*1 #200
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
inp.system.ad = dict(inputs=dict(t = 5.5),
                     input_type="point_forces",
                     grad_type="jacrev", #"jacrev", #value
                     objective_fun="pmean",
                     objective_var="ra",
                     objective_args=dict(nodes=(35,), components=(0,1,2,3,4,5),
                                         t=(inp.system.t[-1],))
                     )

sol11 = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)

#############
epsilon = 1e-2
inp.system.ad = dict(inputs=dict(t = 5.5),
                     input_type="point_forces",
                     grad_type="value", #"jacrev", #value
                     objective_fun="pmean",
                     objective_var="ra",
                     objective_args=dict(nodes=(35,), components=(0,1,2,3,4,5),
                                         t=(inp.system.t[-1],))
                     )
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/ShardingMC_AD12")

sol12 = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)
inp.system.ad = dict(inputs=dict(t = 5.5+epsilon),
                     input_type="point_forces",
                     grad_type="value", #"jacrev", #value
                     objective_fun="pmean",
                     objective_var="ra",
                     objective_args=dict(nodes=(35,), components=(0,1,2,3,4,5),
                                         t=(inp.system.t[-1],))
                     )
inp.driver.sol_path = pathlib.Path(
    f"{results_path}/ShardingMC_AD13")

sol13 = feniax.feniax_shardmain.main(input_dict=inp, device_count=8)

jac = (sol12.staticsystem_s1.objective - sol13.staticsystem_s1.objective) / epsilon
