# Wing Sail Plane

Verifying the nonlinear structural dynamics on a clamped configuration.

## Load modules

``` python
import plotly.express as px
import pyNastran.op4.op4 as op4
import matplotlib.pyplot as plt
import pdb
import datetime
import os
import shutil
REMOVE_RESULTS = True
#   for root, dirs, files in os.walk('/path/to/folder'):
#       for f in files:
#           os.unlink(os.path.join(root, f))
#       for d in dirs:
#           shutil.rmtree(os.path.join(root, d))
# 
if os.getcwd().split('/')[-1] != 'results':
    if not os.path.isdir("./figs"):
        os.mkdir("./figs")
    if REMOVE_RESULTS:
        if os.path.isdir("./results"):
            shutil.rmtree("./results")
    if not os.path.isdir("./results"):
        print("***** creating results folder ******")
        os.mkdir("./results")
    os.chdir("./results")
```

``` {#PYTHONMODULES .python}
import pathlib
import pickle
import jax.numpy as jnp
import jax
import pandas as pd
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader
from tabulate import tabulate
```

## Run cases

``` python

import time

TIMES_DICT = dict()
SOL = dict()
CONFIG = dict()

def run(input1, **kwargs):
    jax.clear_caches()
    label = kwargs.get('label', 'default')
    t1 = time.time()
    config =  configuration.Config(input1)
    sol = feniax.feniax_main.main(input_obj=config)
    t2 = time.time()
    TIMES_DICT[label] = t2 - t1      
    SOL[label] = sol
    CONFIG[label] = config

def save_times():
    pd_times = pd.DataFrame(dict(times=TIMES_DICT.values()),
                            index=TIMES_DICT.keys())
    pd_times.to_csv("./run_times.csv")
```

``` {#wingSP .python}

wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
```

``` {#wingSP_dts .python}

wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
dts = [round(1./ eigenvals[i]**0.5, 2) for i in [5,15,30,50,100]]
```

WSP1

``` {#WSP1 .python}


wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
inp.fem.num_modes = 5
inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

run(inp, label=name)
```

WSP2

``` {#WSP2 .python}


wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
inp.fem.num_modes = 15
inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

run(inp, label=name)
```

WSP3

``` {#WSP3 .python}


wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
inp.fem.num_modes = 30
inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

run(inp, label=name)
```

WSP4

``` {#WSP4 .python}


wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
inp.fem.num_modes = 50
inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

run(inp, label=name)
```

WSP4alpha05

``` {#WSP4alpha05 .python}


wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
inp.fem.num_modes = 50
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 0.5 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 0.5 * 6e5,  0., 0.]
                                                     ]
inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

run(inp, label=name)
```

WSP4alpha15

``` {#WSP4alpha15 .python}

wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
inp.fem.num_modes = 50
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1.5 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1.5 * 6e5,  0., 0.]
                                                     ]
inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
inp.driver.sol_path = pathlib.Path(
    f"./{name}")
run(inp, label=name)
```

WSP5

``` {#WSP5 .python}


wingSP_folder = feniax.PATH / "../examples/wingSP"
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path(f'{wingSP_folder}/FEM/')
eigenvals = jnp.load(inp.fem.folder / "eigenvals.npy")
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5", max_steps=30000) #"rk4")
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
dts = [round(1./ eigenvals[i]**0.5, 6) for i in [5,15,30,50,100]]
print(dts)
inp.fem.num_modes = 100
inp.systems.sett.s1.dt = round(1./ eigenvals[inp.fem.num_modes]**0.5, 6)
inp.driver.sol_path = pathlib.Path(
    f"./{name}")

run(inp, label=name)
```

## Postprocessing

### Plotting functions

``` python
print(f"Format for figures: {figfmt}")
def fig_out(name, figformat=figfmt, update_layout=None):
    def inner_decorator(func):
        def inner(*args, **kwargs):
            fig = func(*args, **kwargs)
            if update_layout is not None:
                fig.update_layout(**update_layout)
            fig.show()
            figname = f"figs/{name}.{figformat}"
            fig.write_image(f"../{figname}", scale=6)
            return fig, figname
        return inner
    return inner_decorator


def fig_background(func):

    def inner(*args, **kwargs):
        fig = func(*args, **kwargs)
        # if fig.data[0].showlegend is None:
        #     showlegend = True
        # else:
        #     showlegend = fig.data[0].showlegend

        fig.update_xaxes(
                       titlefont=dict(size=20),
                       tickfont = dict(size=20),
                       mirror=True,
                       ticks='outside',
                       showline=True,
                       linecolor='black',
            #zeroline=True,
        #zerolinewidth=2,
            #zerolinecolor='LightPink',
                       gridcolor='lightgrey')
        fig.update_yaxes(tickfont = dict(size=20),
                       titlefont=dict(size=20),
                       zeroline=True,
                       mirror=True,
                       ticks='outside',
                       showline=True,
                       linecolor='black',
                       gridcolor='lightgrey')
        fig.update_layout(plot_bgcolor='white',
                          yaxis=dict(zerolinecolor='lightgrey'),
                          showlegend=True, #showlegend,
                          margin=dict(
                              autoexpand=True,
                              l=0,
                              r=0,
                              t=2,
                              b=0
                          ))
        return fig
    return inner
```

``` python

def subplots_wtips2(fun, *args, **kwargs):

    fig1 = fun(*args, dim=0, **kwargs)
    fig2 = fun(*args, dim=1, **kwargs)
    fig3 = fun(*args, dim=2, **kwargs)
    fig3.update_xaxes(title=None)
    fig2.update_xaxes(title=None)
    fig = make_subplots(rows=3, cols=1, horizontal_spacing=0.135, vertical_spacing=0.1,
                        # specs=[[{"colspan": 2}, None],
                        #       [{}, {}]]
                        )
    for i, f3i in enumerate(fig3.data):
        fig.add_trace(f3i,
                      row=1, col=1
                      )
    for i, f1i in enumerate(fig1.data):
        f1inew = f1i
        f1inew.showlegend = False          
        fig.add_trace(f1inew,
                      row=2, col=1
                      )
    for i, f2i in enumerate(fig2.data):
        f2inew = f2i
        f2inew.showlegend = False          
        fig.add_trace(f2inew,
                      row=3, col=1
                      )

    fig.update_xaxes(fig2.layout.xaxis,row=2, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig2.layout.yaxis,row=2, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_xaxes(fig1.layout.xaxis,row=3, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig1.layout.yaxis,row=3, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_xaxes(fig3.layout.xaxis,row=1, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_yaxes(fig3.layout.yaxis,row=1, col=1,titlefont=dict(size=15),
                       tickfont = dict(size=15))
    fig.update_layout(plot_bgcolor='white',
                      yaxis=dict(zerolinecolor='lightgrey'),
                      showlegend=True, #showlegend,
                      margin=dict(
                          autoexpand=True,
                          l=0,
                          r=0,
                          t=2,
                          b=0
                          ))
    fig.update_layout(legend=dict(x=0.81, y=1))
    #fig.update_layout(showlegend=False,row=2, col=1)
    # fig.update_layout(showlegend=False,row=2, col=2)
    #fig.update_layout(fig1.layout)
    return fig

@fig_background
def wsp_wingtip(sol_list, dim, labels=None, nast_load=None, axes=None,
                modes = [5, 15, 30, 50, 100],scale = 1./28.8):

    fig = None
    colors=["red", "darkgreen",
            "steelblue", "magenta", "blue"]
    dash = ['dash', 'dot', 'dashdot']

    for i, si in enumerate(sol_list):
        x, y = putils.pickIntrinsic2D(si.data.dynamicsystem_s1.t,
                                      si.data.dynamicsystem_s1.ra,
                                      fixaxis2=dict(node=23, dim=dim))
        if i != len(sol_list) - 1:
          fig = uplotly.lines2d(x, (y - y[0]) * scale, fig,
                                dict(name=f"NMROM-{modes[i]}",
                                     line=dict(color=colors[i],
                                               dash=dash[i % 3])
                                     ),
                                dict())
        else:
          fig = uplotly.lines2d(x, (y - y[0]) * scale, fig,
                                dict(name=f"NMROM-{modes[i]}",
                                     line=dict(color=colors[i])
                                     ),
                                dict())              
    if nast_load is not None:
        fig = uplotly.lines2d(t_wsp[nast_load], u_wsp[nast_load,:,-4, dim]* scale, fig,
                              dict(name="FullFE-NL",
                                   line=dict(color="black",
                                             dash="dash")
                                   ))
        fig = uplotly.lines2d(t_wspl[nast_load], u_wspl[nast_load,:,-4, dim]* scale, fig,
                              dict(name="FullFE-Lin",
                                   line=dict(color="orange",
                                             #dash="dash"
                                             )
                                   ))
    dim_dict = {0:'x', 1:'y', 2:'z'}
    if axes is None:
        fig.update_yaxes(title=r'$\Large u_%s / l$'%dim_dict[dim])
        fig.update_xaxes(range=[0, 15], title='$\large time \; [s]$')
    else:
        fig.update_yaxes(range=axes[1], title=r'$\large u_%s / l$'%dim_dict[dim])
        fig.update_xaxes(range=axes[0], title='$\large time \; [s]$')

    return fig

@fig_background
def wsp_rootload(sol_list, dim,
                 labels = ['0.5', '1.', '1.5'], nodei=2, scale = 1e-3):

    fig = None
    colors=["red", "darkgreen",
            "steelblue", "magenta", "blue"]
    dash = ['dash', 'dot', 'dashdot']

    for i, si in enumerate(sol_list):
        x, y = putils.pickIntrinsic2D(si.data.dynamicsystem_s1.t,
                                      si.data.dynamicsystem_s1.X2,
                                      fixaxis2=dict(node=nodei, dim=dim))
        if i != len(sol_list) - 1:
          fig = uplotly.lines2d(x, (y - y[0]) * scale, fig,
                                dict(name=f"NMROM-{labels[i]}",
                                     line=dict(color=colors[i],
                                               dash=dash[i % 3])
                                     ),
                                dict())
        else:
          fig = uplotly.lines2d(x, (y - y[0]) * scale, fig,
                                dict(name=f"NMROM-{labels[i]}",
                                     line=dict(color=colors[i])
                                     ),
                                dict())              
    dim_dict = {0:'x', 1:'y', 2:'z'}
    fig.update_yaxes(title=r'$\large f_%s \; [MN/m]$'%(dim+1))
    fig.update_xaxes(range=[0, 10], title='$\large time \; [s]$')

    return fig

def subplots_wsp(sol_list, labels=None, nast_load=None, axes=None):

    fig1 = wsp_wingtip(sol_list, 0, labels, nast_load, axes)
    fig2 = wsp_wingtip(sol_list, 1, labels, nast_load, axes)
    fig3 = wsp_wingtip(sol_list, 2, labels, nast_load, axes)
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=1, vertical_spacing=5,
                        specs=[[{"colspan": 2}, None],
                               [{}, {}]])
    for i, f3i in enumerate(fig3.data):
        fig.add_trace(f3i,
                      row=1, col=1
                      )
    for i, f1i in enumerate(fig1.data):
        f1inew = f1i
        f1inew.showlegend = False          
        fig.add_trace(f1inew,
                      row=2, col=1
                      )
    for i, f2i in enumerate(fig2.data):
        f2inew = f2i
        f2inew.showlegend = False          
        fig.add_trace(f2inew,
                      row=2, col=2
                      )

    fig.update_xaxes(fig1.layout.xaxis,row=2, col=1)
    fig.update_yaxes(fig1.layout.yaxis,row=2, col=1)
    fig.update_xaxes(fig2.layout.xaxis,row=2, col=2)
    fig.update_yaxes(fig2.layout.yaxis,row=2, col=2)
    fig.update_xaxes(fig3.layout.xaxis,row=1, col=1)
    fig.update_yaxes(fig3.layout.yaxis,row=1, col=1)
    fig.update_layout(plot_bgcolor='white',
                      yaxis=dict(zerolinecolor='lightgrey'),
                      showlegend=True, #showlegend,
                      margin=dict(
                          autoexpand=True,
                          l=0,
                          r=0,
                          t=2,
                          b=0
                          ))
    #fig.update_layout(showlegend=False,row=2, col=1)
    # fig.update_layout(showlegend=False,row=2, col=2)
    #fig.update_layout(fig1.layout)
    return fig


def fn_wspError(sol_list):
    error_dict = dict()
    for i, si in enumerate(sol_list):
        for di in range(3):
            x, y = putils.pickIntrinsic2D(si.data.dynamicsystem_s1.t,
                                          si.data.dynamicsystem_s1.ra,
                                          fixaxis2=dict(node=23, dim=di))
            yinterp = jnp.interp(t_wsp, x, y)
            ynastran = u_wsp[0,:,-4, di] + y[0]
            n = 10000
            error = jnp.linalg.norm((yinterp[1,:n] - ynastran[:n]) / ynastran[:n]) / len(ynastran[:n])
            label = f"M{i}x{di}"
            error_dict[label] = error

    return error_dict

@fig_background
def fn_wspPloterror(error):

    loads = [200, 250, 300, 400, 480, 530]
    num_modes = [5, 15, 30, 50, 100]
    ex1 = [error[f'M{i}x0'] for i in range(5)]
    ex2 = [error[f'M{i}x1'] for i in range(5)]
    ex3 = [error[f'M{i}x2'] for i in range(5)]
    fig = None
    fig = uplotly.lines2d(num_modes, ex1, fig,
                              dict(name="Error - x1",
                                   line=dict(color="red")
                                   ),
                              dict())
    fig = uplotly.lines2d(num_modes, ex2, fig,
                              dict(name="Error - x2",
                                   line=dict(color="green")
                                   ),
                              dict())
    fig = uplotly.lines2d(num_modes, ex3, fig,
                              dict(name="Error - x3",
                                   line=dict(color="black")
                                   ),
                              dict())

    fig.update_yaxes(type="log", tickformat= '.0e')
    return fig

@fig_background
def fn_wspPloterror3D(time, error):

    fig = None
    fig = uplotly.lines2d(time, error, fig,
                              dict(name="Error",
                                   line=dict(color="blue")
                                   ),
                              dict())

    fig.update_yaxes(type="log", tickformat= '.0e', nticks=7)
    fig.update_layout(
                      #height=700,
                      xaxis_title=r'$\Large time [s]$',
                      yaxis_title=r'$\Large \epsilon$')
    return fig
```

### Load Nastran data

``` python

import pathlib
import pickle
import jax.numpy as jnp
import jax
import pandas as pd
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader
from tabulate import tabulate

examples_path = pathlib.Path("../../../../examples")

####### wingSP ###########
wingSP_folder = examples_path / "wingSP"
nastran_path = wingSP_folder / "NASTRAN/"
nas_wspl = op2reader.NastranReader(op2name=(nastran_path / "wing_109d.op2"),
                                   bdfname=(nastran_path / "wing_109b.bdf"))
nas_wspl.readModel()
t_wspl, u_wspl = nas_wspl.displacements()  
# ###
nas_wsp = op2reader.NastranReader(op2name=(nastran_path / "wing400d.op2"),
                                   bdfname=(nastran_path / "wing_109b.bdf"))
nas_wsp.readModel()
t_wsp, u_wsp = nas_wsp.displacements()
wsp_error3d = jnp.load(examples_path/ "wingSP/wsp_err.npy")
```

### Structural verification of a representative configuration

1.  Large-amplitude nonlinear dynamics

    This test case demonstrates the accuracy of the NMROM approach for
    dynamic geometrically-nonlinear calculations. The right wing of Fig.
    is considered and dynamic nonlinear simulations are carried out and
    compared to commercial solutions of the full FE model. A force is
    applied at the wing tip with a triangular loading profile, followed
    by a sudden release of the applied force to heavily excite the wing.
    The force profile is given in Fig. [1](#fig:ramping_load). The
    applied force is then
    $f_{tip} = \alpha \textup{\pmb{f}}_{max} f(0.05, 4)$ with
    $\textup{\pmb{f}}_{max} = [-2\times 10^5, 0., 6\times 10^5]$ where
    $\alpha$ has been set to $1$.

    ```{=org}
    #+name: fig:ramping_load
    ```
    ```{=org}
    #+caption: Ramping load profile for dynamic simulation of representative wing
    ```
    ```{=org}
    #+attr_latex: :width 0.6\textwidth
    ```
    [file:./figs_ext/ramping_load.pdf](./figs_ext/ramping_load.pdf) The
    dynamic response is presented in Fig. [1](#fig:wsp_3d), where
    results have been normalised with the wing semi-span ($l=28.8$ m).
    As expected, linear analysis over-predicts vertical displacements
    and does not capture displacements in the $x$ and $y$ directions.
    NMROMs were built with 5, 15, 30, 50 and 100 modes. A Runge-Kutta
    four is used to march the equation in time with time steps
    corresponding to the inverse of the largest eigenvalue in the NMROM,
    i.e. $\Delta t = [27.34, 6.62, 2.49, 1.27, 0.575] \times 10^{-3}$ s.

    ``` {#WSPsubplots .python}
    sol_wsp= [solution.IntrinsicReader(f"./WSP{i}") for i in [1,2,4]] #range(1,6)]
    # fig, figname = fig_out(name)(wsp_wingtip)(sol_wsp, dim=0, labels=None, nast_load=0)
    #fig = subplots_wsp(sol_wsp, labels=None, nast_load=0)
    #figname
    fig, figname = fig_out(name, update_layout=dict(legend=dict(x=0.13, y=0.9385,
        font=dict(size= 10))))(subplots_wtips2)(wsp_wingtip, sol_wsp, labels=None, nast_load=0, modes=[5,15,50])
    figname
    ```

    ![Span-normalised wing-tip displacements in the response to an
    initially ramped load](figs/WSPsubplots.png){#fig:wsp_3d}

    As in the previous example, the 3D shape of the model is retrieved
    and compared against the full nonlinear dynamic solution, as
    illustrated in Fig. [2](#wsp_3d) (Nastran solution in yellow and
    NMROM with 50 modes in blue). The times at positive and negative
    peaks are displayed. Even though a wing of such characteristics
    would never undergo in practice this level of deformations, these
    results further support the viability of the methodology to solve
    highly geometrically nonlinear dynamics, on complex models and with
    minimal computational effort.

    ![Snapshots of wing 3D dynamic response comparing NMROM (blue) and
    NLFEM3D (yellow)](./figs_ext/WSP_3D-front.png){#wsp_3d}

    Next we look at the differences of the dynamic simulations with the
    same metric employed above that now evolves in time. Integration
    errors accumulate and discrepancies grow with time but still remain
    small. In fact the differences between MSC Nastran and our dynamic
    solvers are comparable to the static example with the highest load
    (around the $5\times 10^{-5}$ mark). Both cases displaying maximum
    deformations around 25\\% of the wing semi-span.

    ``` {#WSP_error .python}
    wsp_error = fn_wspError(sol_wsp)
    wsp_error_time = jnp.linspace(0,15,10001)
    fig, figname = fig_out(name, update_layout=dict(showlegend=False, margin=dict(
                                  autoexpand=True,
                                  l=0,
                                  r=5,
                                  t=2,
                                  b=0)))(fn_wspPloterror3D)(wsp_error_time,wsp_error3d)
    figname
    ```

    ```{=org}
    #+name: WSP_error
    ```
    ```{=org}
    #+caption: $\ell^2$ norm per node differences between full FE nonlinear solution and NMROM with 50 modes
    ```
    ```{=org}
    #+attr_latex: :width 0.7\textwidth
    ```
    ```{=org}
    #+results: WSP_error
    ```
    [file:figs/WSP_error.pdf](figs/WSP_error.pdf)

    An impressive reduction of computational time is achieved by our
    solvers as highlighted in Table [1](#table:WSP_times). The nonlinear
    response of the full model took 1 hour 22 minutes, which is over two
    orders of magnitude slower than the NMROM with 50 modes resolution,
    which proved very accurate. The significant increase in
    computational effort when moving from a solution with 50 modes to
    100 modes is due to various factors: vectorised operations are
    limited and the quadratic nonlinearities ultimately lead to
    O($N_m^3$) algorithms; the time-step needs to be decreased for the
    Runge-Kutta integration to remain stable; the additional overheads
    that come with saving and moving larger tensors, from the modal
    shapes, the cubic modal couplings, to the system states (note times
    shown account for all the steps from start to end of the simulation,
    including saving all the data for postprocessing).

    ``` {#WSP_times .python}
    dfruns = pd.read_csv('./run_times.csv',index_col=0).transpose()
    values = ["Time [s]"]
    values += [', '.join([str(round(dfruns[f'WSP{i+1}'].iloc[0], 2)) for i in range(5)])]
    values += [1*60*60 + 22*60]
    values += [33.6]
    header = ["NMROM (modes: 5, 15, 30, 50, 100)"]
    header += ["NASTRAN 400"]
    header += ["NASTRAN 109"]
    # df_sp = pd.DataFrame(dict(times=TIMES_DICT.values()),
    #                         index=TIMES_DICT.keys())

    # df_ = results_df['shift_conm2sLM25']
    # df_ = df_.rename(columns={"xlabel": "%Chord"})
    tabulate([values], headers=header, tablefmt='orgtbl')
    ```

                   NMROM (modes: 5, 15, 30, 50, 100)   NASTRAN 400   NASTRAN 109
      ------------ ----------------------------------- ------------- -------------
      Time \[s\]   2.79, 2.92, 4.85, 12.14, 155.3      4920          33.6

      : Computational times representative wing dynamic solution

2.  Differentiation of dynamic response

    We move now to a novel feature of this work, i.e. the ability to
    compute gradients via automatic differentiation in geometrically
    nonlinear dynamic problems. The maximum root loads occurring in a
    wing subjected to dynamic loads is a good test case as it can be a
    critical metric in sizing the aircraft wings, especially high-aspect
    ratio ones. Thus we look at the variation of the maximum z-component
    of the vertical internal forces as a function of $\alpha$ in the
    loading profile of Fig. [1](#fig:ramping_load). Effectively, the
    slope of the loading increases with $\alpha$. Table
    [2](#table:AD_WSP) shows the derivatives computed using FD with an
    epsilon of $10^{-4}$ and AD in reverse-mode on the example with 50
    modes resolution. In this case the FD needed tweaking of epsilon
    while application of AD was straight forward with no need for
    checkpoints and took around three times the speed of a single
    calculation.

      $\alpha$   $f(\alpha)$ \[KN/m\]   $f'(\alpha)$ (AD)   $f'(\alpha)$ (FD)
      ---------- ---------------------- ------------------- -------------------
      0.5        1706.7                 3587.71             3587.77
      1.0        3459.9                 3735.26             3735.11
      1.5        5398.7                 3957.81             3958.31

      : AD verification structural dynamic problem

    It is worth noting the high frequency dynamics excited after the
    ramping load is suddenly released. In fact in the $z$-component of
    the wing-tip evolution in Fig. [3](#fig:wsp_adz) we can see a
    maximum tip displacement of 4.36 m, 7.91 m and 10.83 m, for
    $\alpha = 0.5, 1, 1.5$ i.e smaller than the proportional linear
    response. On the other hand, in Fig. [4](#fig:wsp_adx2) the
    evolution of the root loads show a response with much higher
    frequencies and the maximum occurs in the free dynamical response of
    the wing, which is higher as we increase $\alpha$.

    ``` {#WSP_adz .python}
    sol_wspz= [solution.IntrinsicReader(f"./WSP{i}") for i in ["4alpha05",
                                                              "4",
                                                              "4alpha15"]] #range(1,6)]
    # fig, figname = fig_out(name)(wsp_wingtip)(sol_wsp, dim=0, labels=None, nast_load=0)
    #fig = subplots_wsp(sol_wsp, labels=None, nast_load=0)
    #figname
    fig, figname = fig_out(name, update_layout=dict(legend=dict(x=0.13, y=0.3,
        font=dict(size= 16))))(wsp_wingtip)(sol_wspz, dim=2, modes=["0.5","1","1.5"], axes=[[0,10],None])
    figname
    ```

    ![Span-normalised wing-tip $z$-displacement for load profiles with
    $\alpha = 0.5, 1, 1.5$](figs/WSP_adz.png){#fig:wsp_adz}

    ``` {#WSP_adx2 .python}

    fig, figname = fig_out(name, update_layout=dict(legend=dict(x=0.13, y=0.941,
        font=dict(size= 16))))(wsp_rootload)(sol_wspz, dim=2, scale=1e-6)
    figname
    ```

    ![Wing root loads, vertical force](figs/WSP_adx2.png){#fig:wsp_adx2}
