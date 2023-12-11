name=[]
import pathlib
import plotly.express as px
import pickle
import jax.numpy as jnp
import pandas as pd
import numpy as np
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main
import fem4inas.plotools.uplotly as uplotly
import fem4inas.plotools.utils as putils
import fem4inas.preprocessor.solution as solution
import fem4inas.unastran.op2reader as op2reader

with open (fem4inas.PATH / "../examples/ArgyrisBeam" / "argyris_new.pickle", 'rb') as fp:
    argypickle  = pickle.load(fp)
argypickle['c'][10][0] = argypickle['c'][10][0][6:]
argypickle['c'][10][1] = argypickle['c'][10][1][6:]

name="ArgyrisBeamPlot"

figname = "figs/ArgyrisBeam.png"
sol_argyf = solution.IntrinsicReader("./ArgyrisBeam")
config_argy =  configuration.Config.from_file("./ArgyrisBeam/config.yaml")
icomp = putils.IntrinsicStructComponent(config_argy.fem)
icomp.add_solution(sol_argyf.data.staticsystem_s1.ra)
colors = px.colors.qualitative.G10
loads = ["Load: 3.7 KN",
         "Load: 12.1 KN" ,
         "Load: 17.5 KN",
         "Load: 39.3 KN",
         "Load: 61. KN",
         "Load: 94.5 KN",
         "Load: 120 KN"]
settline = list()
settmark = list()
annotations = list()
annotations.append(dict(#xref='paper',
    x=icomp.map_mra['ref1'][-1,0]+1,
    y=icomp.map_mra['ref1'][-1,1]+3,
    xanchor='right', yanchor='middle',
    text="Load: 0. KN",
    font=dict(family='Arial',
              size=14),
    showarrow=False))

for i in range(8):
      line_settings=dict(mode="lines+markers",
                         #marker_symbol="218",
                         line=dict(color=colors[i],
                                   width=2.5)
                          )
      marker_settings=dict(mode="markers",
                           marker_symbol="17",
                           marker=dict(color=colors[i+1],
                                       size=10)
                          )

      settline.append(line_settings)
      settmark.append(marker_settings)

      if i < 7:
          annotations.append(dict(#xref='paper',
              x=float(icomp.map_mra[i+2][-1,0]-3),
              y=float(icomp.map_mra[i+2][-1,1]+1),
              xanchor='right', yanchor='middle',
              text=loads[i],
              font=dict(family='Arial',
                        size=14),
              showarrow=False))
# plot intrinsic solution
fig = uplotly.render2d_multi(icomp,
                               scatter_settings=settline)
# plot data from Argyris
fig = uplotly.iterate_lines2d([pi[0] for i, pi in enumerate(argypickle['c']) if (i % 2 ==0)],
                              [pi[1] for i, pi in enumerate(argypickle['c']) if (i % 2 ==0)],
                              scatter_settings=settmark,
                              fig=fig)
fig.update_layout(margin=dict(
      autoexpand=True,
      l=0,
      r=1.5,
      t=1.5,
      b=0
))

fig.update_xaxes(range=[-25, 105],title='x [cm]',
                 titlefont=dict(size=16),
                 tickfont = dict(size=16),
                 mirror=True,
                 ticks='outside',
                 showline=True,
                 linecolor='black',
                 gridcolor='lightgrey'
)
fig.update_yaxes(range=[-85, 65],title='y [cm]',
                 titlefont=dict(size=16),
                 tickfont = dict(size=16),
                 mirror=True,
                 ticks='outside',
                 showline=True,
                 linecolor='black',
                 gridcolor='lightgrey'
)
fig.update_layout(showlegend=False,plot_bgcolor='white',
                  annotations=annotations)
fig.show()
fig.write_image(f"../{figname}")
figname
