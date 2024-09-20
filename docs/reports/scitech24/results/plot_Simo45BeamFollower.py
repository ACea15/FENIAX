name="Simo45FollowerPlot"

import pathlib
import plotly.express as px
import pickle
import jax.numpy as jnp
import pandas as pd
import numpy as np
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import feniax.plotools.uplotly as uplotly
import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution
import feniax.unastran.op2reader as op2reader

simo45beam_folder = feniax.PATH / "../examples/Simo45Beam"
u1=pd.read_csv(simo45beam_folder / "validationdata/u1.csv", names=["f","disp"])
u2=pd.read_csv(simo45beam_folder / "validationdata/u2.csv", names=["f","disp"])
u3=pd.read_csv(simo45beam_folder / "validationdata/u3.csv", names=["f","disp"])
config_simo45f =  configuration.Config.from_file("./Simo45Follower/config.yaml")

figname = "figs/s45follower.png"
sol_s45f = solution.IntrinsicReader("./Simo45Follower")
icomp = putils.IntrinsicStructComponent(config_simo45f.fem)
#icomp.add_solution(config_simo45f.fem.X.T)
icomp.add_solution(sol_s45f.data.staticsystem_s1.ra)
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
loads = [f"Load: {li} N" for li in config_simo45f.systems.mapper['s1'].xloads.follower_interpolation[0][1:]]
for i in range(11):
    line_settings=dict(mode="lines+markers",
                       #marker_symbol="218",
                       line=dict(color='navy',#colors[i],
                                 width=3.5)
                        )
    settline.append(line_settings)
    if i < 10:
        annotations.append(dict(#xref='x',
            x=icomp.map_mra[i+2][-1,0]-5.3,
            y=icomp.map_mra[i+2][-1,1],
            z=icomp.map_mra[i+2][-1,2],
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
    eye=dict(x=-0.3, y=2.5, z=1.)
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
                  scene=dict(
                  annotations=annotations))

fig.show()
fig.write_image(f"../{figname}")
figname

name="Simoverificationfollower"
figname = f"figs/{name}.png"
f = list(config_simo45f.systems.mapper['s1'].xloads.follower_interpolation[0][1:])
u1i = (sol_s45f.data.staticsystem_s1.ra[:,0,-1] -
    config_simo45f.fem.X[-1,0])
u2i = (sol_s45f.data.staticsystem_s1.ra[:,1,-1] -
    config_simo45f.fem.X[-1,1])
u3i = (sol_s45f.data.staticsystem_s1.ra[:,2,-1] -
    config_simo45f.fem.X[-1,2])
settline = [dict(mode="lines",
                 line=dict(color="navy",
                           width=2.5),
               name="u1"
                        ),
            dict(mode="lines",
                 line=dict(color="navy",
                           width=2.5,
                           dash='dot'),
               name="u2"
                        ),
            dict(mode="lines",
                 line=dict(color="navy",
                           width=2.5,
                           dash='dash'),
               name="u3"
                        )
          ]

settmark = [dict(mode="markers",
                 marker_symbol="circle-open",
                 marker=dict(color="navy",
                             size=10),
               name="u1-ref"
                        ),
          dict(mode="markers",
                 marker_symbol="square-open",
                 marker=dict(color="navy",
                             size=10),
             name="u2-ref"
                        ),
          dict(mode="markers",
                 marker_symbol="star-open",
                 marker=dict(color="navy",
                             size=10),
             name="u3-ref"
                        )
          ]

fig = uplotly.iterate_lines2d([jnp.hstack([0,f]), jnp.hstack([0,f]), jnp.hstack([0,f])],
                              [jnp.hstack([0,u1i]), jnp.hstack([0,u2i]), jnp.hstack([0,u3i])],
                              scatter_settings=settline,
                              fig=None)

fig = uplotly.iterate_lines2d([u1.f, u2.f[0::2], u3.f],
                              [u1.disp, u2.disp[0::2], u3.disp],
                              scatter_settings=settmark,
                              fig=fig)
fig.update_xaxes(title='Load [N]',
                 titlefont=dict(size=16),
                 tickfont = dict(size=16))
fig.update_yaxes(title='Disp [m]', tickfont = dict(size=16),
                 titlefont=dict(size=16))
fig.update_layout(legend=dict(font=dict(size=15)),
    margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))

#fig.update_xaxes(range=[-25, 105])
#fig.update_yaxes(range=[-85, 65])
#fig.update_layout(showlegend=False,
#                  annotations=annotations)
fig.show()
fig.write_image(f"../{figname}")
figname
