
Simo\'s flying spaguetti
========================

Verifying the nonlinear structural dynamics on a free-free
configuration.

Load modules
------------

``` {.python}
import plotly.express as px
import pyNastran.op4.op4 as op4
import matplotlib.pyplot as plt
import pdb
import datetime
import os
import shutil
REMOVE_RESULTS = False
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
import plotly.express as px
import pickle
import jax.numpy as jnp
import jax
import pandas as pd
import numpy as np
import pathlib
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.uplotly as uplotly
import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader
import feniax.plotools.nastranvtk.bdfdef as bdfdef
from tabulate import tabulate
examples_folder = pathlib.Path.cwd() / "../../../../examples"    
```

Run cases
---------

``` {.python}
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

-   Models run on this exercise:

``` {#rrb .python}
inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'0': None}
inp.fem.folder = examples_folder / 'SimoFSpaguetti/FEMshell25'
inp.fem.eig_type = "scipy"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.bc1 = 'free'
inp.systems.sett.s1.t1 = 10.
inp.systems.sett.s1.dt = 5e-4
inp.systems.sett.s1.solver_library = "runge_kutta" #"runge_kutta" #"diffrax" #
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4") # "rk4" "Dopri5"
inp.systems.sett.s1.xloads.dead_forces = True
```

### 25 Nodes

25 node discretization of asets

1.  2D~150m~

    ``` {#rrb2d_25n_150m .python}
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'0': None}
    inp.fem.folder = examples_folder / 'SimoFSpaguetti/FEMshell25'
    inp.fem.eig_type = "scipy"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.bc1 = 'free'
    inp.systems.sett.s1.t1 = 10.
    inp.systems.sett.s1.dt = 5e-4
    inp.systems.sett.s1.solver_library = "runge_kutta" #"runge_kutta" #"diffrax" #
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.solver_settings = dict(solver_name="rk4") # "rk4" "Dopri5"
    inp.systems.sett.s1.xloads.dead_forces = True
    inp.driver.sol_path = pathlib.Path(
        f"{results_path}/{name}")
    inp.fem.num_modes = 150
    inp.systems.sett.s1.xloads.dead_points = [[24, 0],
                                              [24, 5]]
    inp.systems.sett.s1.xloads.x = [0., 2.5, 2.5+1e-6, 15.5]
    inp.systems.sett.s1.xloads.dead_interpolation = [[8., 8., 0., 0.],
                                                     [-80., -80., 0., 0.]
                                                     ]
    run(inp, label=name)
    ```

2.  3D~150m~

    ``` {#rrb3d_25n_150m .python}
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'0': None}
    inp.fem.folder = examples_folder / 'SimoFSpaguetti/FEMshell25'
    inp.fem.eig_type = "scipy"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.bc1 = 'free'
    inp.systems.sett.s1.t1 = 10.
    inp.systems.sett.s1.dt = 5e-4
    inp.systems.sett.s1.solver_library = "runge_kutta" #"runge_kutta" #"diffrax" #
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.solver_settings = dict(solver_name="rk4") # "rk4" "Dopri5"
    inp.systems.sett.s1.xloads.dead_forces = True
    inp.driver.sol_path = pathlib.Path(
        f"{results_path}/{name}")
    inp.fem.num_modes = 150
    inp.systems.sett.s1.xloads.dead_points = [[24, 0],
                                            [24, 4],
                                            [24, 5]]
    inp.systems.sett.s1.xloads.x = [0., 2.5, 5., 20.5]
    inp.systems.sett.s1.xloads.dead_interpolation = [[0., 20., 0., 0.],
                                                   [0., 100., 0., 0.],
                                                   [0., -200., 0., 0.]
                                                   ]
    run(inp, label=name)
    ```

### 50 nodes

50 node discretization of asets

1.  2D~300m~

    ``` {#rrb2d_50n_300m .python}
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'0': None}
    inp.fem.folder = examples_folder / 'SimoFSpaguetti/FEMshell25'
    inp.fem.eig_type = "scipy"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.bc1 = 'free'
    inp.systems.sett.s1.t1 = 10.
    inp.systems.sett.s1.dt = 5e-4
    inp.systems.sett.s1.solver_library = "runge_kutta" #"runge_kutta" #"diffrax" #
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.solver_settings = dict(solver_name="rk4") # "rk4" "Dopri5"
    inp.systems.sett.s1.xloads.dead_forces = True
    inp.driver.sol_path = pathlib.Path(
        f"{results_path}/{name}")
    inp.fem.num_modes = 300
    inp.systems.sett.s1.xloads.dead_points = [[24, 0],
                                              [24, 5]]
    inp.systems.sett.s1.xloads.x = [0., 2.5, 2.5+1e-6, 15.5]
    inp.systems.sett.s1.xloads.dead_interpolation = [[8., 8., 0., 0.],
                                                     [-80., -80., 0., 0.]
                                                     ]
    run(inp, label=name)
    ```

2.  3D~300m~

    ``` {#rrb3d_50n_300m .python}
    inp = Inputs()
    inp.engine = "intrinsicmodal"
    inp.fem.connectivity = {'0': None}
    inp.fem.folder = examples_folder / 'SimoFSpaguetti/FEMshell25'
    inp.fem.eig_type = "scipy"
    inp.driver.typeof = "intrinsic"
    inp.simulation.typeof = "single"
    inp.systems.sett.s1.solution = "dynamic"
    inp.systems.sett.s1.bc1 = 'free'
    inp.systems.sett.s1.t1 = 10.
    inp.systems.sett.s1.dt = 5e-4
    inp.systems.sett.s1.solver_library = "runge_kutta" #"runge_kutta" #"diffrax" #
    inp.systems.sett.s1.solver_function = "ode"
    inp.systems.sett.s1.solver_settings = dict(solver_name="rk4") # "rk4" "Dopri5"
    inp.systems.sett.s1.xloads.dead_forces = True
    inp.driver.sol_path= pathlib.Path(
        f"./{name}")
    inp.fem.num_modes = 300
    inp.systems.sett.s1.xloads.dead_points = [[24, 0],
                                            [24, 4],
                                            [24, 5]]
    inp.systems.sett.s1.xloads.x = [0., 2.5, 5., 20.5]
    inp.systems.sett.s1.xloads.dead_interpolation = [[0., 20., 0., 0.],
                                                   [0., 100., 0., 0.],
                                                   [0., -200., 0., 0.]
                                                   ]
    run(inp, label=name)
    ```

    ``` {.python}
    save_times()
    ```

Postprocessing
--------------

### Plotting functions

``` {.python}
scale_quality = 6
print(f"Format for figures: {figfmt}")
print(f"Image quality: {scale_quality}")  
def fig_out(name, figformat=figfmt, update_layout=None):
    def inner_decorator(func):
        def inner(*args, **kwargs):
            fig = func(*args, **kwargs)
            if update_layout is not None:
                fig.update_layout(**update_layout)
            fig.show()
            figname = f"figs/{name}.{figformat}"
            fig.write_image(f"../{figname}", scale=scale_quality)
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

@fig_background
def plot_Xtime(t, X,  dim, X0=[0.,0.,0., 0.,0.,0.], labels=None, node=150,
            scale=100./33.977, x_range=[0,4]):
    fig=None
    if labels is None:
        labels = list(range(X))
    colors = ["steelblue", "steelblue", "green", "green"]
    dashes = ["solid", "dash"]*2
    for i, Xi in enumerate(X):
        x1, y1 = putils.pickIntrinsic2D(t,
                                        Xi,
                                        fixaxis2=dict(node=node, dim=dim))
        if i == 0:
            y10=y1[0]
            print(y10)
        fig = uplotly.lines2d(x1, (y1 - X0[dim])/y10 * scale, fig,
                              dict(name=f"NMROM-{labels[i]}",
                                   line=dict(color=colors[i],
                                             dash=dashes[i])
                               ))

    dim_dict = {0:'x', 1:'y', 2:'z', 3:r'\theta_x', 4:r'\theta_y', 5:r'\theta_z'}      
    fig.update_yaxes(title=r'$\large u_{%s}$'%dim_dict[dim])
    fig.update_xaxes(range=x_range, title='time [s]')
    return fig

@fig_background
def plot_Xcomponents(X,  dim1, dim2, labels=None, node=150,
            scale1=1, scale2=1):
    fig=None
    if labels is None:
        labels = list(range(X))
    colors = ["steelblue", "steelblue", "green", "green"]
    dashes = ["solid", "dash"]*2
    fig = uplotly.lines2d([1], [1], fig,
                          dict(name=None,
                               showlegend=False,
                               #line=dict(color=colors[i]),
                               marker=dict(symbol="star", color="red",size=16)
                               ),
                          dict())

    for i, Xi in enumerate(X):
        x1, y1 = putils.pickIntrinsic2D(Xi,
                                        Xi,
                                        fixaxis1=dict(node=node, dim=dim1),
                                        fixaxis2=dict(node=node, dim=dim2)
                                        )
        if i == 0:
            x10 = x1[0]
            y10 = y1[0]
        if i == 2:
            x10 = x1[0]
            y10 = y1[0]

        fig = uplotly.lines2d(x1 / x10 * scale1, y1 / y10 * scale2, fig,
                              dict(name=f"NMROM-{labels[i]}",
                                   line=dict(color=colors[i],
                                             dash=dashes[i])
                               ))

    dim_dict = {0:r'\hat{F}_x', 1:'\hat{F}_y', 2:'\hat{F}_z',
                3:'\hat{M}_x', 4:'\hat{M}_y', 5:'\hat{M}_x'}
    fig.update_xaxes(title=r'$\large %s$' %dim_dict[dim1])
    fig.update_yaxes(title=r'$\large %s$' %dim_dict[dim2])
    return fig

def subplots_X(fun, *args, **kwargs):

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
    fig.update_layout(legend=dict(x=0.1, y=1))
    #fig.update_layout(showlegend=False,row=2, col=1)
    # fig.update_layout(showlegend=False,row=2, col=2)
    #fig.update_layout(fig1.layout)
    return fig

@fig_background
def plot_ffb_times(modes, y1, y2, label1, label2):

    fig = None
    fig = uplotly.lines2d(modes, y1, fig,
                              dict(name=label1,
                                   line=dict(color="blue")
                                   ),
                              dict())

    fig = uplotly.lines2d(modes, y2, fig,
                          dict(name=label2,
                               line=dict(color="red")
                               ),
                          dict())          
    fig.update_yaxes(type="log", exponentformat="power",
                     #tickformat= '.0e',
                     nticks=8)
    fig.update_layout(legend=dict(x=0.7, y=0.95),
                      height=650,
                      xaxis_title='Num. modes',
                      yaxis_title='Computational times [s]')
    return fig

@fig_background
def plot_ffb_error(modes, y1, label1):

    fig = None
    fig = uplotly.lines2d(modes, y1, fig,
                              dict(name=label1,
                                   line=dict(color="blue")
                                   ),
                              dict())
    fig.update_yaxes(type="log", exponentformat="power", #tickformat= '.0e',
                     nticks=8)
    fig.update_layout(showlegend=False,
                      #height=800,
                      xaxis_title='Num. modes',
                      yaxis_title='Cg error')
    return fig

def get_trimaoa_ti(ti, q, omega, phi1):
    q2= q[ti, 0:-1]
    q0i = - q2[2:]/ omega[2:]
    q0 = jnp.hstack([q2[:2], q0i])
    X0 = jnp.tensordot(phi1, q0, axes=(0, 0))
    return X0[4,0]

def get_trimaoa(q, omega, phi1):

    q_len = len(q)
    aoa = [get_trimaoa_ti(i,
                         q,
                         omega,
                         phi1) for i in range(q_len)]
    return jnp.array(aoa)

def get_trimelevator(q, ti=None):
    q_len = len(q)
    if ti is None:
        q_elevator = [q[i, -1] for i in range(q_len)]
    else:
        q_elevator= q[ti, -1]
    return jnp.array(q_elevator)

@fig_background
def plot_trimaoa(loads, alphas, labels):

    rad2grad = 180./3.141592
    colors = ['blue', 'green', 'red', 'black', 'orange', 'yellow']
    symbols = ["circle-open", "square", "diamond-open", "circle"]
    fig = None
    for i, (alphai, labeli) in enumerate(zip(alphas, labels)):
        fig = uplotly.lines2d(loads, rad2grad*alphai, fig,
                                  dict(name=labeli,
                                       line=dict(color=colors[i]),
                                       marker=dict(symbol=symbols[i], size=16)
                                       ),
                                  dict())

    #fig.update_yaxes(type="log", tickformat= '.0e', nticks=8)
    fig.update_layout(#showlegend=False,
                      #height=800,
                      yaxis_range=[0,19],
                      xaxis_range=[0.5,4.1],
                      legend=dict(x=0.1, y=0.9),
                      xaxis_title='Loads [n-g]',
                      yaxis_title=r'$Angle [^o]$')
    return fig
```

``` {.python}
@fig_background
def plot_ffb_times(modes, y1, y2, label1, label2):

    fig = None
    fig = uplotly.lines2d(modes, y1, fig,
                              dict(name=label1,
                                   line=dict(color="blue")
                                   ),
                              dict())

    fig = uplotly.lines2d(modes, y2, fig,
                          dict(name=label2,
                               line=dict(color="red")
                               ),
                          dict())          
    fig.update_yaxes(type="log", exponentformat="power",
                     #tickformat= '.0e',
                     nticks=8)
    fig.update_layout(legend=dict(x=0.7, y=0.95),
                      height=650,
                      xaxis_title='Num. modes',
                      yaxis_title='Computational times [s]')
    return fig

@fig_background
def plot_ffb_error(modes, y1, label1):

    fig = None
    fig = uplotly.lines2d(modes, y1, fig,
                              dict(name=label1,
                                   line=dict(color="blue")
                                   ),
                              dict())
    fig.update_yaxes(type="log", exponentformat="power", #tickformat= '.0e',
                     nticks=8)
    fig.update_layout(showlegend=False,
                      #height=800,
                      xaxis_title='Num. modes',
                      yaxis_title='Cg error')
    return fig
```

### Unsupported dynamics of very flexible structure

This example exemplifies the ability of our solvers to turn a generic
linear free-free finite-element model into a fully nonlinear solution
that accounts for the rigid-body dynamics coupled with large elastic
deformations. It has already been presented in
[cite:&PALACIOS2019](cite:&PALACIOS2019), though the novelties
introduced herein are the new optimised implementation that can run on
accelerators and the approach to recover the full 3D state from the
reduced model. The beam version of this structure was first studied by
Simo and Vu-Quoc [cite:&SIMO1988](cite:&SIMO1988) and has served to
verify several implementations of nonlinear beam dynamics with rigid
body motions [cite:&HESSE2014](cite:&HESSE2014). A straight structure of
constant square cross section (side = 3, wall thickness = 3/10) is built
consisting of 784 shell elements linked to 50 spanwise nodes via
interpolation elements as depicted in Fig. [1](#fig:FFS) together with
the material properties and two types of loading: firstly, a dead-force
in the x-direction and dead-moment in the z-direction that yield a
planar motion in the x-y plane; and secondly, the addition of a moment
in the y-direction which produces a three dimensional motion.

[file:figs\_ext/ffbw10.pdf](figs_ext/ffbw10.pdf)

The free-flying evolution of the 3D model is shown in Fig.
[1](#fig:FFB_2D) for the planar motion and Fig. [1](#fig:FFB_3D) for the
loads giving rise to the full 3D deformations. It worth remarking the
latter motion also exhibits large torsional deformations which are
combined with the also large bending displacements and rigid-body modes.

[file:figs\_ext/FFB\_2D3.pdf](figs_ext/FFB_2D3.pdf)

[file:figs\_ext/FFB\_3D3.pdf](figs_ext/FFB_3D3.pdf)

Because the applied load is a dead force we can track the position of
the center-of-gravity (CG) analytically as a verification exercise.
Furthermore, the highly nonlinear nature of this problem makes it a good
example to showcase the strength of accelerators for large problems and
to gain insights as to when it might be better to deploy the codes in
standard CPUs instead. Therefore we perform a sweep with the number of
modes kept in the solution from 50 to 300, which determines the size of
the system to be solved. The full modal basis is employed at 300 modes
and due to the nonlinear cubic term this entails operations of the order
of $O(10^7)$ at every time step of the solution, making it a good case
for accelerators. The increase in the number of modes also restricts the
incremental time step used in the explicit solver to preserve stability.
Table [1](#table:FFB_times) shows both computational time and CG error
for the planar case in two scenarios: linking the integration time-step
to the largest eigenvalue $\lambda$ in the solution $dt=\lambda^{-0.5}$;
and fixing it to $dt=10^{-3}$. The error metric is defined as the L-2
norm divided by the time steps. Computations have been carried out in
AMD EPYC 7742 CPU processors and Nvidia GPU RTX 6000 at the Imperial
College cluster.

::: {#table:FFB_times}
  Nmodes   CPU (time/err)   GPU (time/err)   CPU (time/err)   GPU (time/err)
  -------- ---------------- ---------------- ---------------- ----------------
  50       7/1.3e-1         9.9/1.3e-1       42/2.1e-2        58/2.1e-2
  100      9.3/5.7e-2       10.4/5.7e-2      184/1.2e-2       65/1.2e-2
  150      34/2.2e-2        14/2.2e-2        287/5.6e-3       67/5.6e-3
  200      79/2e-3          22/2e-3          421/7.2e-4       76/7.2e-4
  250      474/5.3e-4       38/5.3e-4        893/2.7e-4       94/2.7e-4
  300      1869/2.54e-5     111/2.54e-5      1869/2.54e-5     111/2.54e-5

  : FFB computational times in seconds and CG error
:::

Fig. [2](#fig:FFBtimes2) and [3](#fig:FFBerror2) illustrate the times
and error results in the table for the second case with fixed time step.
The gain in performance from the GPU is is more impressive the larger
the system to solve, and for the full modal basis the CPU takes more
than 31 minutes versus the less than 2 minutes in the GPU. Computational
times in the 3D problem are similar and the error on the CG position is
slightly higher: for the 300 modes case, the error is $6.9e-5$ versus
the $2.54e-5$ of the planar case.

``` {#FFBtimes1 .python}
modes = [50,100,150,200,250,300]
err1 = [1.3e-1, 5.7e-2, 2.2e-2, 2e-3, 5.3e-4, 2.54e-5]
err2 = [2.1e-2, 1.2e-2, 5.6e-3, 7.2e-4, 2.7e-4, 2.54e-5]
gpu_times1 = [9.9, 10.4, 14, 22, 38, 111]
cpu_times1 = [7, 9.3, 34, 79, 474, 1869]
gpu_times2 = [58, 65, 67, 76, 94, 111]
cpu_times2 = [42, 184, 287, 421, 893, 1869]
fig, figname = fig_out(name)(plot_ffb_times)(modes, gpu_times1, cpu_times1, "GPU", "CPU")
figname
```

![Performance CPU vs GPU comparison in free-flying structure (variable
time step)](figs/FFBtimes1.png "FFBtimes")

``` {#FFBtimes2 .python}
fig, figname = fig_out(name)(plot_ffb_times)(modes, gpu_times2, cpu_times2, "GPU", "CPU")
figname
```

![Performance CPU vs GPU comparison in free-flying structure (fixed time
step)](figs/FFBtimes2.png "FFBtimes2")

``` {#FFBerror1 .python}
fig, figname = fig_out(name)(plot_ffb_error)(modes, err1, 'L2-norm Error')
#figname
```

[file:]()

``` {#FFBerror2 .python}
fig, figname = fig_out(name, update_layout=dict(showlegend=False))(plot_ffb_error)(modes, err2, 'L2-norm Error')
figname
```

![Error metric CG position for planar
case](figs/FFBerror2.png "FFBerror2")
