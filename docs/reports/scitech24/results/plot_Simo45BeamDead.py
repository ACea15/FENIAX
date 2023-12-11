name="Simo45DeadPlot"
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

figname = "figs/s45dead.png"
sol_s45d = solution.IntrinsicReader("./Simo45Dead")
config_simo45d = configuration.Config.from_file("./Simo45Dead/config.yaml")
icomp = putils.IntrinsicStructComponent(config_simo45d.fem)
#icomp.add_solution(config_simo45f.fem.X.T)
icomp.add_solution(sol_s45d.data.staticsystem_s1.ra)
settline = list()
annotations = list()
colors = px.colors.qualitative.Dark24
annotations.append(dict(#xref='paper',
    x=icomp.map_mra['ref1'][-1,0]-13,
    y=icomp.map_mra['ref1'][-1,1],
    xanchor='right', yanchor='middle',
    text="Load: 0 N",
    font=dict(family='Arial',
              size=12),
    showarrow=False))
loads = [f"Load: {li} N" for li in config_simo45d.systems.mapper['s1'].xloads.dead_interpolation[0][1:]]
for i in range(11):
    line_settings=dict(mode="lines+markers",
                       #marker_symbol="218",
                       line=dict(color=colors[i],
                                 width=3.5),
                       marker=dict(size=5)
                        )
    settline.append(line_settings)
    if i < 10:
        annotations.append(dict(#xref='x',
            x=icomp.map_mra[i+2][-1,0]-8,
            y=icomp.map_mra[i+2][-1,1],
            z=icomp.map_mra[i+2][-1,2]+3,
            #xanchor='right', yanchor='middle',
            text=loads[i],
            font=dict(family='Arial',
                      size=12),
            showarrow=False))

#fig = uplotly.render3d_struct(icomp)
fig = uplotly.render3d_multi(icomp,
                             scatter_settings=settline)
fig.update_traces(marker=dict(size=1.5))
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5*2, y=0.75, z=0.5)
)
fig.update_layout(autosize=True,
                  width=1200,
                  height=1200,
                  scene_camera=camera,
                  margin=dict(
                      autoexpand=True,
                      l=0,
                      r=0,
                      t=0,
                      b=0,
                      pad=0
                  ),
                  showlegend=False,
                  # scene=dict(
                  # annotations=annotations)
                  )
fig.update_xaxes(range=[-10, 35])
fig.show()
fig.write_image(f"../{figname}")
figname
