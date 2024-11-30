# [[file:modelgeneration.org::*Run][Run:1]]
import pathlib
import time
import jax.numpy as jnp
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.reconstruction as reconstruction
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
inp.driver.typeof = "intrinsic"
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
inp.system.xloads.modalaero_forces = True
inp.system.xloads.x = [0.,1.]
inp.system.t = [1/3, 2/3, 1]#[0.25, 0.5, 0.75, 1]
inp.system.aero.c_ref = c_ref
inp.system.aero.u_inf = u_inf # a0(7000) =312
inp.system.aero.rho_inf = rho_inf
inp.system.aero.poles = f"./AERO/{Poles_file}.npy"
inp.system.aero.A = f"./AERO/{Ahh_file}.npy"
inp.system.aero.Q0_rigid = f"./AERO/Qhx{label_gaf}.npy"
inp.system.aero.qalpha = [[0.,  0., 0, 0, 0, 0],
                          [0.,  6 * np.pi / 180, 0, 0, 0, 0]] # interpolation: x=0 => qalpha=0
                                                              # x=1 => qalpha = 4   
inp.driver.sol_path = pathlib.Path(
    "./results/manoeuvre2")  
config =  configuration.Config(inp)
solstatic1 = feniax.feniax_main.main(input_obj=config)
# Run:1 ends here

# [[file:modelgeneration.org::3Dstatic][3Dstatic]]
rintrinsic, uintrinsic = reconstruction.rbf_based(
        nastran_bdf="./NASTRAN/BUG103.bdf",
        X=config.fem.X,
        time=range(len(inp.system.t)),
        ra=solstatic1.staticsystem_sys1.ra,
        Rab=solstatic1.staticsystem_sys1.Cab,
        R0ab=solstatic1.modes.C0ab,
        vtkpath=inp.driver.sol_path / "paraview/solstatic1/bug",
        plot_timeinterval=1,
        plot_ref=False,
        tolerance=1e-3,
        size_cards=8,
        rbe3s_full=False,
        ra_movie=None)
# 3Dstatic ends here
