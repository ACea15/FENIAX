# [[file:modelgeneration.org::DiscreteLoads][DiscreteLoads]]
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
num_modes = 300
device_count = 1
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
    f"{results_path}/ShardDiscreteLoads1")

inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "static"
inp.system.solver_library = "diffrax"
inp.system.solver_function = "newton"
inp.system.solver_settings = dict(rtol=1e-6,
                                  atol=1e-6,
                                  max_steps=100,
                                  norm="linalg_norm",
                                  kappa=0.01)
inp.system.xloads.follower_forces = True
inp.system.xloads.x = [0, 1, 2, 3, 4, 5]
inp.system.t = [0.5, 1, 1.5, 2, 2.5, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5.]
lz1 = 5e4  * 0.5
lz2 = 9e4  * 0.5
lz3 = 2e5  * 0.5
lz4 = 4e5  * 0.5
lz5 = 5e5  * 0.5 
lx1 = lz1 * 5 
lx2 = lz2 * 5
lx3 = lz3 * 5
lx4 = lz4 * 5
lx5 = lz5 * 5
ly1 = lz1 * 7
ly2 = lz2 * 7
ly3 = lz3 * 7
ly4 = lz4 * 7
ly5 = lz5 * 7

# rwing: 14-35
# lwing: 40-61
                      # [[[node_i, component_j]..(total_forces per run)],...(parallel forces)[[node_i, component_j]..]]      
inputforces = dict(follower_points=[[[35, 0], [61, 0], [35, 1], [61, 1]],
                                    [[35, 1], [61, 1], [35, 0], [61, 0]],
                                    [[35, 2], [61, 2], [35, 0], [61, 0]],
                                    [[35, 3], [61, 3], [35, 0], [61, 0]],
                                    [[35, 4], [61, 4], [35, 0], [61, 0]],
                                    [[35, 5], [61, 5], [35, 0], [61, 0]],
                                    [[35, 0], [61, 0], [35, 4], [61, 4]],
                                    [[35, 2], [61, 2], [35, 4], [61, 4]],
                                    ],
                   # [[[0,...interpolation points]..(total_forces per run)],...(parallel forces)[[0,...]..]]
                   follower_interpolation= [[[0., lx1, lx2, lx3, lx4, lx5], 
                                             [0., lx1, lx2, lx3, lx4, lx5],
                                             [0., 0., 0., 0., 0., 0.],
                                             [0., 0., 0., 0., 0., 0.]], 
                                            [[0., ly1, ly2, ly3, ly4, ly5], 
                                             [0., ly1, ly2, ly3, ly4, ly5],
                                             [0., 0., 0., 0., 0., 0.],
                                             [0., 0., 0., 0., 0., 0.]], 
                                            [[0., lz1, lz2, lz3, lz4, lz5], 
                                             [0., lz1, lz2, lz3, lz4, lz5],
                                             [0., 0., 0., 0., 0., 0.],
                                             [0., 0., 0., 0., 0., 0.]], 
                                            [[0., lz1, lz2, lz3, lz4, lz5], 
                                             [0., lz1, lz2, lz3, lz4, lz5],
                                             [0., 0., 0., 0., 0., 0.],
                                             [0., 0., 0., 0., 0., 0.]], 
                                            [[0., lx1, lx2, lx3, lx4, lx5], 
                                             [0., lx1, lx2, lx3, lx4, lx5],
                                             [0., 0., 0., 0., 0., 0.],
                                             [0., 0., 0., 0., 0., 0.]], 
                                            [[0., lx1, lx2, lx3, lx4, lx5], 
                                             [0., lx1, lx2, lx3, lx4, lx5],
                                             [0., 0., 0., 0., 0., 0.],
                                             [0., 0., 0., 0., 0., 0.]], 
                                            [[0., lz1, lz2, lz3, lz4, lz5], 
                                             [0., lz1, lz2, lz3, lz4, lz5], 
                                             [0., lx1, lx2, lx3, lx4, lx5], 
                                             [0., lx1, lx2, lx3, lx4, lx5]
                                             ], 
                                            [[0., lz1, lz2, lz3, lz4, lz5], 
                                             [0., lz1, lz2, lz3, lz4, lz5], 
                                             [0., lx1, lx2, lx3, lx4, lx5], 
                                             [0., lx1, lx2, lx3, lx4, lx5]]
                                            ]
                 )  
inp.system.shard = dict(input_type="pointforces",
                        inputs=inputforces)
sol = feniax.feniax_shardmain.main(input_dict=inp, device_count=device_count)
# DiscreteLoads ends here
