---
author: Alvaro Cea and Rafael Palacios
title: JAX-based Aeroelastic Simulation Engine for Differentiable
  Aircraft Dynamics
---

```{=org}
#+setupfile: ./config.org
```
# House keeping

# Load modules

``` python
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
import pathlib
import plotly.express as px
import pickle
import jax.numpy as jnp
import jax
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.uplotly as uplotly
import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader
import feniax.plotools.nastranvtk.bdfdef as bdfdef
from tabulate import tabulate
```

# Plotting

## Helper functions

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
    fig.update_yaxes(type="log", tickformat= '.0e', nticks=8)
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
    fig.update_yaxes(type="log", tickformat= '.0e', nticks=8)
    fig.update_layout(showlegend=False,
                      #height=800,
                      xaxis_title='Num. modes',
                      yaxis_title='Cg error')
    return fig

@fig_background
def xrf1_wingtip2(sol1, sol2, dim, labels=None,nast_scale=None, nast_load=None):
    scale = 1./33.977
    fig=None
    x1, y1 = putils.pickIntrinsic2D(sol1.data.dynamicsystem_s1.t,
                                  sol1.data.dynamicsystem_s1.ra,
                                  fixaxis2=dict(node=150, dim=dim))
    x2, y2 = putils.pickIntrinsic2D(sol2.data.dynamicsystem_s1.t,
                                  sol2.data.dynamicsystem_s1.ra,
                                  fixaxis2=dict(node=150, dim=dim))

    fig = uplotly.lines2d(x1[:], (y1[:]-y1[0])*scale, fig,
                          dict(name=f"NMROM-G{labels[0]}",
                               line=dict(color="orange")
                               ))
    fig = uplotly.lines2d(x2[1:], (y2[:-1]-y2[0])*scale, fig,
                          dict(name=f"NMROM-G{labels[1]}",
                               line=dict(color="steelblue")
                               ))

    if nast_scale is not None:
        offset = 0. #u111m[nast_load[0],0,-1, dim]
        fig = uplotly.lines2d(t111m[nast_load[0]], (u111m[nast_load[0],:,-1, dim] - offset)*nast_scale*scale, fig,
                              dict(name=f"Lin. FE-G{labels[0]}",
                                   line=dict(color="black",
                                             dash="dash",
                                             width=1.5)
                                   ))
        offset2 = 0. #u111m[nast_load[1],0,-1, dim]
        fig = uplotly.lines2d(t111m[nast_load[1]], (u111m[nast_load[1],:,-1, dim] - offset2)*nast_scale*scale, fig,
                              dict(name=f"Lin. FE-G{labels[1]}",
                                   line=dict(color="red",
                                             dash="dot",
                                             width=1.5)
                                   ))
    dim_dict = {0:'x', 1:'y', 2:'z'}
    fig.update_yaxes(title=r'$\large u_%s / l$'%dim_dict[dim])
    fig.update_xaxes(range=[0, 4], title=r'$\large time \; [s]$')
    return fig

def subplots_wtips(fun, *args, **kwargs):

    fig1 = fun(*args, dim=0, **kwargs)
    fig2 = fun(*args, dim=1, **kwargs)
    fig3 = fun(*args, dim=2, **kwargs)
    fig3.update_xaxes(title=None)
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.135, vertical_spacing=0.1,
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


def subplots_xrf1wtips(sol1, sol2, labels=None, nast_scale=None, nast_load=None):

    fig1 = xrf1_wingtip2(sol1, sol2, 0, labels,nast_scale, nast_load)
    fig2 = xrf1_wingtip2(sol1, sol2, 1, labels,nast_scale, nast_load)
    fig3 = xrf1_wingtip2(sol1, sol2, 2, labels,nast_scale, nast_load)
    fig = make_subplots(rows=3, cols=1, horizontal_spacing=0.135, vertical_spacing=0.1,
                        # specs=[[{"colspan": 2}, None],
                        #       [{}, {}]]
                        )

    # fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.1,
    #                    specs=[[{"colspan": 2}, None],
    #                           [{}, {}]])
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

@fig_background
def xrf1_wingtip4(sol1, sol2, sol3, sol4, dim, labels=None,nast_scale=None, nast_load=None):
    scale = 1./33.977
    fig=None
    x1, y1 = putils.pickIntrinsic2D(sol1.data.dynamicsystem_s1.t,
                                    sol1.data.dynamicsystem_s1.ra,
                                    fixaxis2=dict(node=150, dim=dim))
    x2, y2 = putils.pickIntrinsic2D(sol2.data.dynamicsystem_s1.t,
                                    sol2.data.dynamicsystem_s1.ra,
                                    fixaxis2=dict(node=150, dim=dim))
    x3, y3 = putils.pickIntrinsic2D(sol3.data.dynamicsystem_s1.t,
                                    sol3.data.dynamicsystem_s1.ra,
                                    fixaxis2=dict(node=150, dim=dim))
    x4, y4 = putils.pickIntrinsic2D(sol4.data.dynamicsystem_s1.t,
                                    sol4.data.dynamicsystem_s1.ra,
                                    fixaxis2=dict(node=150, dim=dim))

    fig = uplotly.lines2d(x1[1:], (y1[:-1]-y1[0])*scale, fig,
                          dict(name=f"NMROM-{labels[0]}",
                               line=dict(color="orange",
                                         dash="solid")
                               ))
    fig = uplotly.lines2d(x2[:], (y2[:]-y2[0])*scale, fig,
                          dict(name=f"NMROM-{labels[1]}",
                               line=dict(color="blue", dash="dot")
                               ))
    fig = uplotly.lines2d(x3[:], (y3[:]-y3[0])*scale, fig,
                          dict(name=f"NMROM-{labels[2]}",
                               line=dict(color="red")
                               ))
    fig = uplotly.lines2d(x4[:], (y4[:]-y4[0])*scale, fig,
                          dict(name=f"NMROM-{labels[3]}",
                               line=dict(color="grey", dash="dash")
                               ))

    dim_dict = {0:'x', 1:'y', 2:'z'}
    fig.update_yaxes(title=r'$\large u_%s / l$'%dim_dict[dim])
    fig.update_xaxes(range=[0, 4], title=r'$\large time \; [s]$')
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

def fn_spErrorold(sol_list, config, print_info=True):

  sol_sp= [solution.IntrinsicReader(f"./SP{i}") for i in range(1,6)]
  err = {f"M{i}_L{j}": 0. for i in range(1,6) for j in range(6)}
  for li in range(6): # loads
    for mi in range(1,6):  # modes
      count = 0  
      for index, row in config.fem.df_grid.iterrows():
        r_spn = u_sp[li, row.fe_order,:3] + config.fem.X[index]
        r_sp = sol_sp[mi - 1].data.staticsystem_s1.ra[li,:,index]
        err[f"M{mi}_L{li}"] += jnp.linalg.norm(r_spn - r_sp) #/ jnp.linalg.norm(r_spn)
        # print(f"nas = {r_spn}  ,  {r_sp}")
        count += 1
      err[f"M{mi}_L{li}"] /= count
      if print_info:
          print(f"**** LOAD: {li}, NumModes: {mi} ****")
          print(err[f"M{mi}_L{li}"])
  return err

def fn_spError(sol_list, config, print_info=True):

    sol_sp= [solution.IntrinsicReader(f"./SP{i}") for i in range(1,6)]
    err = {f"M{i}_L{j}": 0. for i in range(1,6) for j in range(6)}
    for li in range(6): # loads
      for mi in range(1,6):  # modes
        count = 0
        r_spn = []
        r_sp = []
        for index, row in config.fem.df_grid.iterrows():
          r_spn.append(u_sp[li, row.fe_order,:3] + config.fem.X[index])
          r_sp.append(sol_sp[mi - 1].data.staticsystem_s1.ra[li,:,index])
          # print(f"nas = {r_spn}  ,  {r_sp}")
          # count += 1
        r_spn = jnp.array(r_spn)
        r_sp = jnp.array(r_sp)        
        err[f"M{mi}_L{li}"] += jnp.linalg.norm(r_spn - r_sp) #/ jnp.linalg.norm(r_spn)
        err[f"M{mi}_L{li}"] /= len(r_sp)
        if print_info:
            print(f"**** LOAD: {li}, NumModes: {mi} ****")
            print(err[f"M{mi}_L{li}"])
    return err

def fn_spWingsection(sol_list, config):

    sol_sp= [solution.IntrinsicReader(f"./SP{i}") for i in range(1,6)]
    r_spn = []
    r_spnl = []
    r_sp = []
    for li in range(6): # loads
      for mi in [4]:#range(1,6):  # modes
        r_spni = []
        r_spnli = []
        r_spi = []
        r_sp0 = []
        for index, row in config.fem.df_grid.iterrows():
          if row.fe_order in list(range(20)):
            r_sp0.append(config.fem.X[index])  
            r_spni.append(u_sp[li, row.fe_order,:3] + config.fem.X[index])
            r_spnli.append(u_spl[li, row.fe_order,:3] + config.fem.X[index])
            r_spi.append(sol_sp[mi - 1].data.staticsystem_s1.ra[li,:,index])
          # print(f"nas = {r_spn}  ,  {r_sp}")
          # count += 1

        r_spn.append(jnp.array(r_spni))
        r_spnl.append(jnp.array(r_spnli))
        r_sp.append(jnp.array(r_spi))
    r_sp0 = jnp.array(r_sp0)
    return r_sp0, r_sp, r_spn, r_spnl

@fig_background
def plot_spWingsection(r0, r, rn, rnl):
    fig = None
    # colors=["darkgrey", "darkgreen",
    #         "blue", "magenta", "orange", "black"]
    # dash = ['dash', 'dot', 'dashdot']
    modes = [5, 15, 30, 50, 100]
    for li in range(6):
      if li == 0:   
          fig = uplotly.lines2d((r[li][:,0]**2 + r[li][:,1]**2)**0.5, r[li][:,2]-r0[:,2], fig,
                                dict(name=f"NMROM",
                                     line=dict(color="blue",
                                               dash="solid")
                                     ),
                                  dict())
          fig = uplotly.lines2d((rn[li][:,0]**2 + rn[li][:,1]**2)**0.5, rn[li][:,2]-r0[:,2], fig,
                                dict(name=f"FullFE-NL",
                                     line=dict(color="black",
                                               dash="dash")
                                     ),
                                dict())
          fig = uplotly.lines2d((rnl[li][:,0]**2 + rnl[li][:,1]**2)**0.5, rnl[li][:,2]-r0[:,2], fig,
                                dict(name=f"FullFE-Lin",
                                     line=dict(color="orange",
                                               dash="solid")
                                     ),
                                dict())

      else:
          fig = uplotly.lines2d((r[li][:,0]**2 + r[li][:,1]**2)**0.5, r[li][:,2]-r0[:,2], fig,
                                dict(showlegend=False,
                                     line=dict(color="blue",
                                               dash="solid")
                                     ),
                                  dict())
          fig = uplotly.lines2d((rn[li][:,0]**2 + rn[li][:,1]**2)**0.5, rn[li][:,2]-r0[:,2], fig,
                                dict(showlegend=False,
                                     line=dict(color="black",
                                               dash="dash")
                                     ),
                                dict())
          fig = uplotly.lines2d((rnl[li][:,0]**2 + rnl[li][:,1]**2)**0.5, rnl[li][:,2]-r0[:,2], fig,
                                dict(showlegend=False,
                                     line=dict(color="orange",
                                               dash="solid")
                                     ),
                                dict())            
    fig.update_yaxes(title=r'$\large u_z [m]$')
    fig.update_xaxes(title=r'$\large S [m]$', range=[6.81,36])
    fig.update_layout(legend=dict(x=0.6, y=0.95),
                      font=dict(size=20))
    # fig = uplotly.lines2d((rnl[:,0]**2 + rnl[:,1]**2)**0.5, rnl[:,2], fig,
    #                       dict(name=f"NASTRAN-101",
    #                            line=dict(color="grey",
    #                                      dash="solid")
    #                                  ),
    #                             dict())
    return fig

@fig_background
def fn_spPloterror(error):

    loads = [200, 250, 300, 400, 480, 530]
    num_modes = [5, 15, 30, 50, 100]
    e250 = jnp.array([error[f'M{i}_L1'] for i in range(1,6)])
    e400 = jnp.array([error[f'M{i}_L3'] for i in range(1,6)])
    e530 = jnp.array([error[f'M{i}_L5'] for i in range(1,6)])
    fig = None
    fig = uplotly.lines2d(num_modes, e250 , fig,
                              dict(name="F = 250 KN",
                                   line=dict(color="red")
                                   ),
                              dict())
    fig = uplotly.lines2d(num_modes, e400, fig,
                              dict(name="F = 400 KN",
                                   line=dict(color="green", dash="dash")
                                   ),
                              dict())
    fig = uplotly.lines2d(num_modes, e530, fig,
                              dict(name="F = 530 KN",
                                   line=dict(color="black", dash="dot")
                                   ),
                              dict())
    fig.update_xaxes(title= {'font': {'size': 20}, 'text': 'Number of modes'})#title="Number of modes",title_font=dict(size=20))
    fig.update_yaxes(title=r"$\Large \epsilon$",type="log", # tickformat= '.1r',
                     tickfont = dict(size=12), exponentformat="power",
                     #dtick=0.2,
                     #tickvals=[2e-2, 1e-2, 7e-3,5e-3,3e-3, 2e-3, 1e-3,7e-4, 5e-4,3e-4, 2e-4, 1e-4, 7e-5, 5e-5]
                     )
    #fig.update_layout(height=650)
    fig.update_layout(legend=dict(x=0.7, y=0.95), font=dict(size=20))

    return fig

@fig_background
def fn_spPloterror3D(error, error3d):

    loads = [200, 250, 300, 400, 480, 530]
    fig = None
    if error is not None:
      fig = uplotly.lines2d(loads, error, fig,
                                dict(name="Error ASET",
                                     line=dict(color="red"),
                                     marker=dict(symbol="square")
                                     ),
                                dict())

    fig = uplotly.lines2d(loads, error3d, fig,
                              dict(name="Error full 3D",
                                   line=dict(color="green")
                                   ),
                              dict())

    fig.update_yaxes(type="log", tickformat= '.0e')
    fig.update_layout(#height=700,
                      # showlegend=False,
                      #legend=dict(x=0.7, y=0.95),
                      xaxis_title='Loading [KN]',
                      yaxis_title=r'$\Large \epsilon$')

    return fig

@fig_background
def plot_spAD(rn, r0):

    loads = [200, 250, 300, 400, 480, 530]
    fig = None
    x = list(range(1,7))
    y = [rn[i-1][-1, 2] - r0[-1,2] for i in x]
    fig = uplotly.lines2d(x, y, fig,
                                dict(#name="Error ASET",
                                     #line=dict(color="red"),
                                     #marker=dict(symbol="square")
                                     ),
                                dict())


    #fig.update_yaxes(type="log", tickformat= '.0e')
    fig.update_layout(#height=700,
                      showlegend=False,
                      xaxis_title=r'$\Large{\tau}$',
                      yaxis_title='Uz [m]'
    )

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

## NASTRAN data

Read data from Nastran simulations

``` python

import pathlib
import plotly.express as px
import pickle
import jax.numpy as jnp
import jax
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.uplotly as uplotly
import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader
import feniax.plotools.nastranvtk.bdfdef as bdfdef
from tabulate import tabulate


examples_path = pathlib.Path("../../../../examples")
####### SailPlane ###########
SP_folder = examples_path / "SailPlane"
#nastran_path = wingSP_folder / "NASTRAN/"

op2model = op2reader.NastranReader(SP_folder / "NASTRAN/static400/run.op2",
                                   SP_folder / "NASTRAN/static400/run.bdf",
                                 static=True)

op2model.readModel()
t_sp, u_sp = op2model.displacements()

op2modell = op2reader.NastranReader(SP_folder / "NASTRAN/static400/run_linear.op2",
                                   SP_folder / "NASTRAN/static400/run_linear.bdf",
                                 static=True)

op2modell.readModel()
t_spl, u_spl = op2modell.displacements()

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
####### XRF1 ###########
nastran_path = examples_path / "XRF1/NASTRAN/146-111/"
nas111 = op2reader.NastranReader(op2name=(nastran_path / "XRF1-146run.op2"))
nas111.readModel()
t111, u111 = nas111.displacements()

nastran_pathm = examples_path / "XRF1/NASTRAN/146-111_081"
nas111m = op2reader.NastranReader(op2name=(nastran_pathm / "XRF1-146run.op2"))
nas111m.readModel()
t111m, u111m = nas111m.displacements()

sp_error3d = jnp.load(examples_path/ "SailPlane/sp_err.npy")
wsp_error3d = jnp.load(examples_path/ "wingSP/wsp_err.npy")
```

# Examples

The cases presented are a demonstration of our solution approach to
manage geometric nonlinearities, the accuracy of the solvers when
compared to full FE simulations, and the computational gains that can be
achieved. All computations are carried out on a single CPU with an
i7-6700 processor of 3.4 GHz clock speed.

## Structural verification of a representative configuration

A representative FE model of a full aircraft without engines is used to
demonstrate a versatile solution that accounts for geometric
nonlinearities in a very efficient manner and only needs modal shapes
and linear FE matrices from a generic FE solver as inputs. Another of
the goals set for this work was to achieve an equally flexible strategy
in the automatic calculation of derivatives across the various solvers
in the code. The structural static and dynamic responses and their
sensitivities with respect to input parameters are verified against MSC
Nastran and finite differences respectively.\
The aircraft's main wing is composed of wing surfaces, rear and front
spars, wing box and ribs with composite materials employed in the
construction. Flexible tail and rear stabiliser are rigidly attached to
the wing (28.8 m of span), as shown in Fig. [1](#fig:SailPlane2). This
aircraft was first used in [cite:&CEA2021a](cite:&CEA2021a) and is a
good test case as it is not very complex yet representative of aircraft
FE models and it is available open source.

![Representative engineless aeroplane structural FE
model](figs_ext/SailPlaneRef.png){#fig:SailPlane2}

A Guyan reduction is employed in the reduction process and Fig.
[2](#fig:modes) illustrates the accuracy of the condensed model by
comparing the 3D modal shapes. No differences can be appreciated for the
first few modes (the lowest frequency corresponding to a bending mode
agrees in both models at $\omega_1=4.995$ rads/s), so we show higher
frequency modes: a high order bending mode ($\omega_{10}=60.887/60.896$
rads/s in full versus reduced models) and a torsional mode
($\omega_{20}=107.967/107.969$ rads/s). This very good preservation of
the full model leads to an excellent accuracy in the static and dynamic
results presented below. It is important to remark this aircraft model
is typical of previous generations airliners and does not feature
high-aspect ratio wings. Therefore while this modelling strategy would
not be suitable for every engineering structure, as long as there is a
dominant dimension and deformations in the other two remain small (as is
the case in high level descriptions of aircraft, bridges or wind
turbines) it has been found to produce very good approximations when
compared with full dimensional solutions.

```{=org}
#+name: fig:modes2
```
```{=org}
#+caption: Full VS reduced order models on the 10th modal shape $\omega_{10}=60.887/60.896$ rads/s
```
```{=org}
#+attr_latex: :width 0.6\textwidth
```
[file:figs_ext/SPM7af2.pdf](figs_ext/SPM7af2.pdf)

```{=org}
#+name: fig:modes
```
```{=org}
#+caption: Full VS reduced order models on the 20th modal shape, $\omega_{20}=107.97$ rads/s
```
```{=org}
#+attr_latex: :width 0.6\textwidth
```
[file:figs_ext/SPM19af2.pdf](figs_ext/SPM19af2.pdf)

### Geometrically nonlinear static response

The static equilibrium of the aircraft under prescribed loads is first
studied with follower loads normal to the wing applied at the tip of
each wing (nodes 25 and 48). The response for an increasing load
stepping of 200, 300, 400, 480 and 530 KN is computed. The snippet of
the inputs and simulation call are given in Listing
`\ref{code:static}`{=latex}.

\begin{listing}[!ht]
\begin{minted}[frame=single]{python}
import feniax.preprocessor.configuration as configuration
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
inp = Inputs()
inp.fem.folder = "./FEM/"
inp.fem.num_modes = 50
inp.systems.sett.s1.solution = "static"
inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_function = "newton"
inp.systems.sett.s1.solver_settings = dict(rtol=1e-6,
                                           atol=1e-6,
                                           max_steps=50,
                                           norm="linalg_norm")
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[25, 2], [48, 2]]
inp.systems.sett.s1.xloads.x = [0, 1, 2, 3, 4, 5, 6]
inp.systems.sett.s1.xloads.follower_interpolation =
[[0., 2e5, 2.5e5, 3.e5, 4.e5, 4.8e5, 5.3e5],
[0., 2e5, 2.5e5, 3.e5, 4.e5, 4.8e5, 5.3e5]]
inp.systems.sett.s1.t = [1, 2, 3, 4, 5, 6]
config =  configuration.Config(inp)
sol = feniax.feniax_main.main(input_obj=config)
\end{minted}
\caption{FENIAX inputs for structural static simulation}
\label{code:static}
\end{listing}

Nonlinear static simulations on the original full model (before
condensation) are also carried out in MSC Nastran and are included. The
interpolation elements in the full FE solver are used to output the
displacements at the condensation nodes for direct comparison with the
NMROM results. Geometric nonlinearities are better illustrated by
representing a sectional view of the wing as in Fig.
[2](#fig:SPWingsection), where deformations in the z-direction versus
the metric $S = \sqrt{x^2+y^2}$ are shown. MSC Nastran linear solutions
(Solution 101) are also included to appreciate more clearly the
shortening and follower force effects in the nonlinear computations.

``` {#SPWingsection .python}
import feniax.preprocessor.configuration as configuration
config = configuration.Config.from_file("SP1/config.yaml")
sol_sp= [solution.IntrinsicReader(f"./SP{i}") for i in range(1,6)]
r_sp0, r_sp, r_spn, r_spnl = fn_spWingsection(sol_sp, config)
fig, figname = fig_out(name)(plot_spWingsection)(r_sp0, r_sp, r_spn, r_spnl)
figname
```

![Static geometrically-nonlinear effects on the aircraft main
wing](figs/SPWingsection.png){#fig:SPWingsection}

The tolerance in the Newton solver was set to $10^{-6}$ in all cases. A
convergence analysis with the number of modes in the solution is
presented in Fig. [3](#SPstatic_convergence). 5, 15, 30, 50, 100 modes
are used to build the corresponding NMROMs. The error metric is defined
as the $\ell^2$ norm divided by the total number of nodes (only the
condenses ones in this case):
$\epsilon = ||u_{NMROM} - u_{NASTRAN}||/N_{nodes}$. It can be seen the
solution with 50 modes already achieves a very good solution even for
the largest load which produces a 25.6$\%$ tip deformation of the wing
semi-span, $b = 28.8$ m. The displacement difference with the full FE
solution at the tip in this case is less than 0.2$\%$.

``` {#SPerror .python}

config = configuration.Config.from_file("SP1/config.yaml")
sol_sp= [solution.IntrinsicReader(f"./SP{i}") for i in range(1,6)]
sp_error = fn_spError(sol_sp, config, print_info=True)
fig, figname = fig_out(name)(fn_spPloterror)(sp_error)
figname
```

![Modal convergence static solution of representative
aircraft](figs/SPerror.png){#SPstatic_convergence}

The 3D structural response has been reconstructed using the approach in
Fig. \[BROKEN LINK: workflow\]. The nodes connected by the interpolation
elements (RBE3s) to the ASET solution are reconstructed first and
subsequently a model with RBFs kernels is used to extrapolate to the
rest of the nodes in the full FE. A very good agreement is found against
the geometrically-nonlinear Nastran solution (SOL 400). Fig.
[4](#SPstatic_3D) shows the overlap in the Nastran solution (in red) and
the NMROM (in blue) for the 530 KN loading.

```{=org}
#+name: SPstatic_3D
```
```{=org}
#+caption: Static 3D solution for a solution with 50 modes and 530 KN loading (Full NASTRAN solution in red versus the NMROM in blue).
```
```{=org}
#+attr_latex: :width 0.7\textwidth
```
![](./figs_ext/SP_3Dloading-front2.png)
![](./figs_ext/SP_3Dloading-side.png) The error metric of this 3D
solution is also assessed in Fig. [4](#fig:SPerror3D), for the solution
with 50 modes. The discrepancy metric is of the same order than the
previously shown at the reduction points. This conveys an important
point, that there is no significant accuracy loss in the process of
reconstructing the 3D solution.

``` {#SPerror3D .python}
sp_error1D = [sp_error[f'M4_L{i}'] for i in range(6)]
# fig, figname = fig_out(name)(fn_spPloterror3D)(sp_error1D, sp_error3d)
fig, figname = fig_out(name,update_layout=dict(showlegend=False))(fn_spPloterror3D)(None, sp_error3d)
figname
```

![Relative error between full FE and NMROM
solutions](figs/SPerror3D.png){#fig:SPerror3D}

Next we compare the computational times for the various solutions
presented in this section in Table [1](#table:SP_times). Computations of
the six load steps in Fig. [2](#fig:SPWingsection) are included in the
assessment. A near 50 times speed-up is achieved with our solvers
compared to Nastran nonlinear solution, which is one of the main
strengths of the proposed method. As expected, the linear static
solution in Nastran is the fastest of the results, given it only entails
solving a linear, very sparse system of equations.

``` {#SP_times .python}
dfruns = pd.read_csv('./run_times.csv',index_col=0).transpose()
values = ["Time [s]"]
values += [', '.join([str(round(dfruns[f'SP{i+1}'].iloc[0], 2)) for i in range(5)])]
values += [5*60 + 45]
values += [1.02]
header = ["NMROM (modes: 5, 15, 30, 50, 100)"]
header += ["NASTRAN 400"]
header += ["NASTRAN 101"]
# df_sp = pd.DataFrame(dict(times=TIMES_DICT.values()),
#                         index=TIMES_DICT.keys())

# df_ = results_df['shift_conm2sLM25']
# df_ = df_.rename(columns={"xlabel": "%Chord"})
tabulate([values], headers=header, tablefmt='orgtbl')
```

               NMROM (modes: 5, 15, 30, 50, 100)   NASTRAN 400   NASTRAN 101
  ------------ ----------------------------------- ------------- -------------
  Time \[s\]   1.97, 2.05, 2.13, 2.17, 2.3         345           1.02

               NMROM (modes: 5, 15, 30, 50, 100)   NASTRAN 400   NASTRAN 101
  ------------ ----------------------------------- ------------- -------------
  Time \[s\]   6.7, 6.63, 6.79, 7.06, 9.55         345           1.02

  : Computational times static solution {#table:SP_times}

### Differentiation of static response

The AD for the static solvers is first verified as follows: the load
stepping shown above becomes a pseudo-time interpolation load such that
a variable $\tau$ controls the amount of loading and we look at the
variation of the wing-tip displacement as a function of this $\tau$. If
$f(\tau=[1, 2, 3, 4, 5, 6]) = [200, 250, 300, 400, 480, 530]$ KN, with a
linear interpolation between points, the derivative of the z-component
of the tip of the wing displacement is computed at
$\tau= 1.5, 3.5, 5.5 $, as show in Fig. [5](#fig:sp_ad) where the
$y$-axis is the tip displacement, $\tau$ is in the $x$-axis and the big
red circles the points where the derivatives are computed (coincident to
the graph slope at those points).

``` {#SP_AD .python}
fig, figname = fig_out(name)(plot_spAD)(r_sp, r_sp0)
#figname
```

```{=org}
#+results: SP_AD
```
[file:]()

```{=org}
#+name: fig:sp_ad
```
```{=org}
#+caption: Static tip displacement with pseudo-time stepping load
```
```{=org}
#+attr_latex: :width 0.5\textwidth
```
<file:figs_ext/sp_ad.pdf> Table [2](#table:SP_AD) shows a very good
agreement against finite-differences (FD) with an epsilon of $10^{-3}$.
Note how the derivative at each of the marked points corresponds
approximately to the slope in the graph at those very points, which
varies as the load steps are not of equal length. And the biggest slope
occurs precisely in between $\tau$ of 4 and 5 when the prescribed
loading undergoes the biggest change from 300 to 400 KN.

  $\tau$   $f(\tau)$ \[m\]   $f'(\tau)$ (AD)   $f'(\tau)$ (FD)
  -------- ----------------- ----------------- -----------------
  1.5      2.81              0.700             0.700
  3.5      4.527             1.344             1.344
  5.5      6.538             0.623             0.623

  : AD verification structural static problem {#table:SP_AD}

### Large-amplitude nonlinear dynamics

This test case demonstrates the accuracy of the NMROM approach for
dynamic geometrically-nonlinear calculations. The right wing of Fig.
[1](#fig:SailPlane2) is considered and dynamic nonlinear simulations are
carried out and compared to commercial solutions of the full FE model. A
force is applied at the wing tip with a triangular loading profile,
followed by a sudden release of the applied force to heavily excite the
wing. The force profile is given in Fig. [5](#fig:ramping_load). The
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
dynamic response is presented in Fig. [5](#fig:wsp_3d), where results
have been normalised with the wing semi-span ($l=28.8$ m). As expected,
linear analysis over-predicts vertical displacements and does not
capture displacements in the $x$ and $y$ directions. NMROMs were built
with 5, 15, 30, 50 and 100 modes. A Runge-Kutta four is used to march
the equation in time with time steps corresponding to the inverse of the
largest eigenvalue in the NMROM, i.e.
$\Delta t = [27.34, 6.62, 2.49, 1.27, 0.575] \times 10^{-3}$ s.

``` {#WSPsubplots .python}
sol_wsp= [solution.IntrinsicReader(f"./WSP{i}") for i in [1,2,4]] #range(1,6)]
# fig, figname = fig_out(name)(wsp_wingtip)(sol_wsp, dim=0, labels=None, nast_load=0)
#fig = subplots_wsp(sol_wsp, labels=None, nast_load=0)
#figname
fig, figname = fig_out(name, update_layout=dict(legend=dict(x=0.13, y=0.9385,
    font=dict(size= 10))))(subplots_wtips2)(wsp_wingtip, sol_wsp, labels=None, nast_load=0, modes=[5,15,50])
figname
```

![Span-normalised wing-tip displacements in the response to an initially
ramped load](figs/WSPsubplots.png){#fig:wsp_3d}

As in the previous example, the 3D shape of the model is retrieved and
compared against the full nonlinear dynamic solution, as illustrated in
Fig. [6](#wsp_3d) (Nastran solution in yellow and NMROM with 50 modes in
blue). The times at positive and negative peaks are displayed. Even
though a wing of such characteristics would never undergo in practice
this level of deformations, these results further support the viability
of the methodology to solve highly geometrically nonlinear dynamics, on
complex models and with minimal computational effort.

![Snapshots of wing 3D dynamic response comparing NMROM (blue) and
NLFEM3D (yellow)](./figs_ext/WSP_3D-front.png){#wsp_3d}

Next we look at the differences of the dynamic simulations with the same
metric employed above that now evolves in time. Integration errors
accumulate and discrepancies grow with time but still remain small. In
fact the differences between MSC Nastran and our dynamic solvers are
comparable to the static example with the highest load (around the
$5\times 10^{-5}$ mark). Both cases displaying maximum deformations
around 25\\% of the wing semi-span.

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

![$\ell^2$ norm per node differences between full FE nonlinear solution
and NMROM with 50 modes](figs/WSP_error.png){#WSP_error}

An impressive reduction of computational time is achieved by our solvers
as highlighted in Table [3](#table:WSP_times). The nonlinear response of
the full model took 1 hour 22 minutes, which is over two orders of
magnitude slower than the NMROM with 50 modes resolution, which proved
very accurate. The significant increase in computational effort when
moving from a solution with 50 modes to 100 modes is due to various
factors: vectorised operations are limited and the quadratic
nonlinearities ultimately lead to O($N_m^3$) algorithms; the time-step
needs to be decreased for the Runge-Kutta integration to remain stable;
the additional overheads that come with saving and moving larger
tensors, from the modal shapes, the cubic modal couplings, to the system
states (note times shown account for all the steps from start to end of
the simulation, including saving all the data for postprocessing).

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

```{=org}
#+results: WSP_times
```
               NMROM (modes: 5, 15, 30, 50, 100)   NASTRAN 400   NASTRAN 109
  ------------ ----------------------------------- ------------- -------------
  Time \[s\]   2.79, 2.92, 4.85, 12.14, 155.3      4920          33.6

  : Computational times representative wing dynamic solution
  {#table:WSP_times}

### Differentiation of dynamic response

We move now to a novel feature of this work, i.e. the ability to compute
gradients via automatic differentiation in geometrically nonlinear
dynamic problems. The maximum root loads occurring in a wing subjected
to dynamic loads is a good test case as it can be a critical metric in
sizing the aircraft wings, especially high-aspect ratio ones. Thus we
look at the variation of the maximum z-component of the vertical
internal forces as a function of $\alpha$ in the loading profile of Fig.
[5](#fig:ramping_load). Effectively, the slope of the loading increases
with $\alpha$. Table [4](#table:AD_WSP) shows the derivatives computed
using FD with an epsilon of $10^{-4}$ and AD in reverse-mode on the
example with 50 modes resolution. In this case the FD needed tweaking of
epsilon while application of AD was straight forward with no need for
checkpoints and took around three times the speed of a single
calculation.

  $\alpha$   $f(\alpha)$ \[KN/m\]   $f'(\alpha)$ (AD)   $f'(\alpha)$ (FD)
  ---------- ---------------------- ------------------- -------------------
  0.5        1706.7                 3587.71             3587.77
  1.0        3459.9                 3735.26             3735.11
  1.5        5398.7                 3957.81             3958.31

  : AD verification structural dynamic problem {#table:AD_WSP}

It is worth noting the high frequency dynamics excited after the ramping
load is suddenly released. In fact in the $z$-component of the wing-tip
evolution in Fig. [8](#fig:wsp_adz) we can see a maximum tip
displacement of 4.36 m, 7.91 m and 10.83 m, for $\alpha = 0.5, 1, 1.5$
i.e smaller than the proportional linear response. On the other hand, in
Fig. [9](#fig:wsp_adx2) the evolution of the root loads show a response
with much higher frequencies and the maximum occurs in the free
dynamical response of the wing, which is higher as we increase $\alpha$.

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

## Aeroelastic dynamic loads on an industrial configuration

The studies presented in this section are based on a reference
configuration developed to industry standards known as XRF1, which is
representative of a long-range wide-body transport airplane. The version
with a wing-tip extension in [@CEA2023] is employed to verify a gust
response against MSC Nastran linear solution of the full FE model. While
the previous results where purely structural, now the dynamic response
to an atmospheric disturbance or gust is computed. This aeroelastic
analysis is a requirement for certification purposes and it is one of
the main drivers in sizing the wings of high aspect ratio wings.
Furthermore, the previous examples showed the advantage of our approach
in terms of computational speed, but other than that results could be
obtained with commercial software. The geometrically nonlinear
aeroelastic response, however, it is not currently available in
commercial solutions that are bounded to linear analysis in the
frequency domain. Other research codes feature those additional physics,
yet are limited to simple models. Thus the added value in the proposed
approach comes at the intersection between the nonlinear physics arising
from large integrated displacements, computational efficiency and the
ability to enhance the models already built for industrial use.\
Fig. [10](#fig:xrf1_modalshapes) shows the reference FE model with three
modal shapes. The FE model contains a total of around 177400 nodes,
which are condensed into 176 active nodes along the reference load axes
through interpolation elements. A Guyan or static condensation approach
is used for the reduction and the error in the natural frequencies
between full and reduced models is kept below 0.1% well beyond the 30th
mode. The aerodynamic model contains $\sim 1,500$ aerodynamic panels.
The simulations are carried out with a modal resolution of 70 modes and
a time step in the Runge-Kutta solver of 0.005.

```{=org}
#+name: fig:xrf1_modalshapes
```
```{=org}
#+caption: Modified XRF1 reference configuration with characteristic modal shapes
```
```{=org}
#+attr_latex: :width 0.8\textwidth
```
[file:figs_ext/xrf1_modalshapes3.pdf](figs_ext/xrf1_modalshapes3.pdf)

### Linear response for low intensity gust

A verification exercise is introduced first by applying two 1-cos gust
shapes at a very low intensity, thus producing small deformations and a
linear response. The flow Mach number is 0.81. A first gust is applied
that we name as G1 of length 67 m and peak velocity 0.141 m/s, and a
second gust, G2, of 165 m and peak velocity of 0.164 m/s. A snippet of
the inputs to the simulation is display in Listing
`\ref{code:dynamic}`{=latex}.

\begin{listing}[!ht]
\begin{minted}[frame=single]{python}
from feniax.preprocessor.inputs import Inputs
\begin{minted}[frame=single]{python}
inp.fem.folder = "./FEM/"
inp.fem.num_modes = 70
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 7.5
inp.systems.sett.s1.tn = 2001
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
inp.systems.sett.s1.xloads.modalaero_forces = True
inp.systems.sett.s1.aero.folder = "./AERO/"
inp.systems.sett.s1.aero.c_ref = 7.271
inp.systems.sett.s1.aero.u_inf = 200.
inp.systems.sett.s1.aero.rho_inf = 1.225
inp.systems.sett.s1.aero.gust_profile = "mc"
inp.systems.sett.s1.aero.gust.intensity = 0.141
inp.systems.sett.s1.aero.gust.length = 67.
\end{minted}
\caption{FENIAX of inputs for dynamic gust simulation}
\label{code:dynamic}
\end{listing}

Fig. [10](#fig:GustXRF12) shows the normalised wing-tip response with
our NMROM that accurately reproduces the linear solution based on the
full FE model.

``` {#GustXRF12 .python}
sol1= solution.IntrinsicReader("./XRF1")
sol2= solution.IntrinsicReader("./XRF2")
fig, figname = fig_out(name)(subplots_wtips2)(xrf1_wingtip2,sol1, sol2, labels=[1,2], nast_scale=0.01, nast_load=[2,6])
figname
```

![Wing-tip response to low intensity
gust](figs/GustXRF12.png){#fig:GustXRF12}

### Nonlinear response for high intensity gust

The gust intensity in the previous section by a factor of 200 in order
to show the effects of geometric nonlinearities that are only captured
by the nonlinear solver. As seen in Fig. [11](#fig:GustXRF34), there are
major differences in the $x$ and $y$ components of the response due to
follower and shortening effects, and a slight reduction in the
$z$-component. These are well known geometrically nonlinear effects that
are added to the analysis with no significant overhead.

``` {#GustXRF34 .python}
sol1= solution.IntrinsicReader("./XRF3")
sol2= solution.IntrinsicReader("./XRF4")
fig, figname = fig_out(name)(subplots_wtips2)(xrf1_wingtip2, sol1, sol2, labels=[1,2], nast_scale=2., nast_load=[2,6])
figname
```

![Wing-tip response to high intensity
gust](figs/GustXRF34.png){#fig:GustXRF34}

Snapshots of the 3D response are reconstructed for the G1 gust using the
method verified above at the time points where tip displacement are
maximum and minimum, i.e. 0.54 and 0.84 seconds. The front and side
views together with the aircraft reference configuration are shown in
Fig. [12](#fig:xrf1gust3D).

```{=org}
#+name: fig:xrf1gust3D
```
```{=org}
#+caption: 3D XRF1 Nonlinear gust response
```
```{=org}
#+attr_latex: :width 1\textwidth
```
[file:figs_ext/xrf1gust3D2.pdf](figs_ext/xrf1gust3D2.pdf)

In previous examples the same Runge-Kutta 4 (RK4) time-marching scheme
is used and now we explore the dynamic solution with other solvers to
assess their accuracy and also their computational performance. Two
explicit ODE solvers, RK4 and Dormand-Prince\'s 5/4 method (labelled S1
and S2), and two implicit, Euler first order and Kvaerno\'s 3/2 method
((labelled S3 and S4)), are compared in Fig. [12](#fig:GustXRF3578). In
order to justify the use of implicit solvers we reduce the time step
from 0.005 to 0.02 seconds, at which point both explicit solvers
diverge. Kvaerno\'s implicit solver remain stable and accurate despite
the larger time step while the Euler implicit method is stable but do
not yield accurate results.

``` {#GustXRF3578 .python}
sol3= solution.IntrinsicReader("./XRF3")
sol5= solution.IntrinsicReader("./XRF5")
sol7= solution.IntrinsicReader("./XRF7")
sol8= solution.IntrinsicReader("./XRF8")

fig, figname = fig_out(name)(subplots_wtips2)(xrf1_wingtip4, sol1=sol3, sol2=sol5, sol3=sol7, sol4=sol8,
                                            labels=["S1","S2","S3","S4"])
figname
```

```{=org}
#+name: fig:GustXRF3578
```
```{=org}
#+caption: Wing-tip response to high intensity gust using implicit solvers
```
```{=org}
#+attr_latex: :width 0.8\textwidth
```
```{=org}
#+results: GustXRF3578
```
[file:]()

The computational times of the different solvers are shown in Table
[5](#table:XRF1_times). The implicit solvers have taken one order of
magnitude more time to run despite the reduction in time step. Therefore
the main take away this is that for moderately large frequency dynamics,
the explicit solvers offer a much efficient solution. The turning point
for using implicit solvers would be when the largest eigenvalue in Eqs.
`\ref{eq2:sol_qs}`{=latex} led to prohibitly small time steps. In terms
of the Nastran solution, we are not showing the whole simulation time
because that would include the time to sample the DLM aerodynamics which
are input into the NMROM as a post-processing step. Instead, the
increase in time when adding an extra gust subcase to an already
existing analysis is shown, i.e. the difference between one simulation
that only computes one gust response and another with two. It is
remarkable that the explicit solvers are faster on the nonlinear
solution than the linear solution by a commercial software. Besides our
highly efficient implementation, the main reason for this might be the
Nastran solution involves first a frequency domain analysis and then an
inverse Fourier transform to obtain the time-domain results.

``` {#XRF1_times .python}
dfruns = pd.read_csv('./run_times.csv',index_col=0).transpose()
values = ["Time [s]"]
values += [', '.join([str(round(dfruns[f'XRF{i}'].iloc[0], 2)) for i in [3,5,7,8]])]
values += [0*60*60 + 1*60 + 21]
header = ["NMROM [S1, S2, S3, S4]" ]
header += ["$\Delta$ NASTRAN 146"]
# df_sp = pd.DataFrame(dict(times=TIMES_DICT.values()),
#                         index=TIMES_DICT.keys())

# df_ = results_df['shift_conm2sLM25']
# df_ = df_.rename(columns={"xlabel": "%Chord"})
tabulate([values], headers=header, tablefmt='orgtbl')
```

```{=org}
#+results: XRF1_times
```
               NMROM \[S1, S2, S3, S4\]       $\Delta$ NASTRAN 146
  ------------ ------------------------------ ----------------------
  Time \[s\]   22.49, 18.94, 273.95, 847.89   81

  : Computational times XRF1 gust solution. {#table:XRF1_times}

### Differentiation of aeroelastic response

Similarly to the examples above, we now verify the AD implementation for
the nonlinear aeroelastic response to the gust $G1$. The sensitivity of
the six components of the wing root loads are computed with respect to
the gust parameters $w_g$ and $L_g$, and the flow parameter
$\rho_{\inf}$. The results are presented in [1](#table:XRF1_AD). A very
good agreement with the finite differences is found with
$\epsilon=10^{-4}$.

```{=org}
#+caption: Automatic differentiation in aeroelastic problem
```
```{=org}
#+name: table:XRF1_AD
```
\begin{table} [h!]
\begin{center}
\begin{tabular}{lllll}
\toprule
 & $w_g$ & $L_g$ & $\rho_{\inf}$ \\
\midrule
$f_1$ (AD) & 12.180 & 6.666 & 477.208 \\
$f_1$ (FD)  & 12.180 & 6.190  & 477.198  \\
\hline
$f_2 (AD)$ & 19.088 & 6.122 & 712.485 \\
$f_2 (FD)$ & 19.088 & 7.045 & 712.514  \\
\hline
$f_3 (AD)$ & 65.574 & 8.218 & 1464.910 \\
$f_3 (FD)$ & 65.574  & 7.813 & 1464.909  \\
\hline
$f_4 (AD)$ & 126.648& 21.598 & 2883.370 \\
$f_4 (FD)$ & 126.648& 19.736  & 2883.371  \\
\hline
$f_5 (AD)$ & 330.759 & 85.224 & 5931.723 \\
$f_5 (FD)$ & 330.759  & 97.188  & 5930.027  \\
\hline
$f_6$ (AD) & 252.128 & 48.423 & 7179.735 \\
$f_6$ (FD) & 252.128  & 14.980  & 7180.023  \\
\bottomrule
\end{tabular}
\end{center}
\end{table}