# [[file:modelgeneration.org::*Run][Run:1]]
import pathlib
import jax.numpy as jnp
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.reconstruction as reconstruction

label_gaf = "Dd1c7F1Scao-50"
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

inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.eig_names = ["eigenvals_50.npy", "eigenvecs_50.npy"]
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./results1manoeuvre")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=100,
                                           norm="linalg_norm",
                                           kappa=0.01)
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.xloads.x = [0.,1.]
inp.systems.sett.s1.t = [0.25, 0.5, 0.75, 1]
inp.systems.sett.s1.aero.c_ref = 3.0
inp.systems.sett.s1.aero.u_inf = 170.
inp.systems.sett.s1.aero.rho_inf = 0.778
inp.systems.sett.s1.aero.poles = f"./AERO/{Poles_file}.npy"
inp.systems.sett.s1.aero.A = f"./AERO/{Ahh_file}.npy"
inp.systems.sett.s1.aero.Q0_rigid = f"./AERO/Qhx{label_gaf}.npy"
inp.systems.sett.s1.aero.qalpha = jnp.array(([0.,  0., 0, 0, 0, 0],
                                           [0.,  6 * jnp.pi / 180, 0, 0, 0, 0]))


config =  configuration.Config(inp)
solstatic1 = feniax.feniax_main.main(input_obj=config)
# Run:1 ends here
plot3D= False
if plot3D:
    reconstruction.rbf_based(
        nastran_bdf="./NASTRAN/BUG103.bdf",
        X=config.fem.X,
        time=range(len(inp.systems.sett.s1.t)),
        ra=solstatic1.staticsystem_s1.ra,
        Rab=solstatic1.staticsystem_s1.Cab,
        R0ab=solstatic1.modes.C0ab,
        vtkpath="./paraview/solstatic1",
        plot_timeinterval=1,
        plot_ref=True,
        tolerance=1e-3,
        size_cards=8,
        rbe3s_full=False,
        ra_movie=None)
