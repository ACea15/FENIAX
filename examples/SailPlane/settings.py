import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "inputs"
inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                           'LWingInner'],
                            FuselageBack=['BottomTail',
                                          'Fin'],
                            RWingInner=['RWingOuter'],
                            RWingOuter=None,
                            LWingInner=['LWingOuter'],
                            LWingOuter=None,
                            BottomTail=['LHorizontalStabilizer',
                                        'RHorizontalStabilizer'],
                            RHorizontalStabilizer=None,
                            LHorizontalStabilizer=None,
                            Fin=None
                            )

inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"

#inp.driver.sol_path = pathlib.Path(
#    f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    f"./results_static")
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton_raphson"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm",
                                           kappa=0.01)
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                           tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_001001'
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]

inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.,
                                                      2e5,
                                                      2.5e5,
                                                      3.e5,
                                                      4.e5,
                                                      4.8e5,
                                                      5.3e5],
                                                     [0.,
                                                      2e5,
                                                      2.5e5,
                                                      3.e5,
                                                      4.e5,
                                                      4.8e5,
                                                      5.3e5]
                                                     ]
inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]

#config =  configuration.Config(inp)

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config, with_comments=True)

sol = fem4inas.fem4inas_main.main(input_obj=config)


CHECK_SOL = False
if CHECK_SOL:
    import fem4inas.plotools.uplotly as uplotly
    import fem4inas.plotools.utils as putils
    import fem4inas.plotools.upyvista as upyvista
    icomp = putils.IntrinsicStructComponent(config.fem)
    icomp.add_solution(sol.staticsystem_s1.ra)
    settings = {}
    fig = uplotly.render3d_multi(icomp,
                                 **settings)

    istruct = putils.IntrinsicStruct(config.fem)
    pl = upyvista.render_wireframe(points=config.fem.X, lines=istruct.lines)
    istruct.add_solution(sol.staticsystem_s1.ra)

    pl = upyvista.render_wireframe(points=config.fem.X, lines=istruct.lines)
    pl.show_grid()
    #pl.view_xy()
    for k, v in istruct.map_ra.items():
        pl = upyvista.render_wireframe(points=v, lines=istruct.lines, pl=pl)
    # import scipy.linalg
    # import numpy as np
    # Ka = np.load("./FEM/Ka.npy")
    # Ma = np.load("./FEM/Ma.npy")
    # w, v = scipy.linalg.eigh(Ka, Ma)

    # save_eigs = True
    # if save_eigs:
    #     np.save("./FEM/eigenvals.npy", w)
    #     np.save("./FEM/eigenvecs.npy", v)


    import fem4inas.plotools.streamlit.intrinsic as sti

    fig = sti.sys_3Dconfiguration0(config)


    icomp = putils.IntrinsicStructComponent(config.fem)
    #breakpoint()
    fig = uplotly.render3d_struct(icomp,
                                  label="ref1",
                                  # scatter_settings=[dict(customdata=ti,
                                  #                        hovertemplate=ti) for ti in template],
                                  update_traces=dict(line=dict(width=1.2,color="navy"),
                                                     marker=dict(size=1.5)))


    icomp = putils.IntrinsicStructComponent(config.fem)
    label = "ref1"
    icomp.add_solution(sol.modes.phi1[0], label_final=label)
    fig = uplotly.render3d_struct(icomp,
                                  label,
                                  **settings)
    fig.show()
