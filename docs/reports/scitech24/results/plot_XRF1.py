name=[]
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

nastran_path = feniax.PATH / "../examples/XRF1/NASTRAN/146-111/"
nas111 = op2reader.NastranReader(op2name=(nastran_path / "XRF1-146run.op2"))
nas111.readModel()
t111, u111 = nas111.displacements()

nastran_pathm = feniax.PATH / "../examples/XRF1/NASTRAN/146-111_081"
nas111m = op2reader.NastranReader(op2name=(nastran_pathm / "XRF1-146run.op2"))
nas111m.readModel()
t111m, u111m = nas111m.displacements()

name="Gust3Plot_x"
gscale = 100./33.977
figname = f"figs/{name}.png"
sol_g3 = solution.IntrinsicReader("./Gust3")
x, y = putils.pickIntrinsic2D(sol_g3.data.dynamicsystem_s1.t,
                              sol_g3.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=150, dim=0))

fig = uplotly.lines2d(x[1:], (y[:-1]-y[0])*gscale, None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ))

fig = uplotly.lines2d(t111m[2], u111m[2,:,-1, 0]*0.01*gscale, fig,
                      dict(name="NASTRAN",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
fig.update_xaxes(range=[0, 4], title='time [s]',
                 titlefont=dict(size=16),
                 tickfont = dict(size=16),
                 mirror=True,
                 ticks='outside',
                 showline=True,
                 linecolor='black',
                 gridcolor='lightgrey')
fig.update_yaxes(title='$\hat{u}_x$', tickfont = dict(size=16),
                 titlefont=dict(size=16),
                 mirror=True,
                 ticks='outside',
                 showline=True,
                 linecolor='black',
                 gridcolor='lightgrey')
fig.update_layout(plot_bgcolor='white',
                  showlegend=False,
                  margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))

fig.show()

fig.write_image(f"../{figname}")
figname

name="Gust3Plot_y"

figname = f"figs/{name}.png"
x, y = putils.pickIntrinsic2D(sol_g3.data.dynamicsystem_s1.t,
                              sol_g3.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=150, dim=1))

fig = uplotly.lines2d(x[1:], y[:-1]-y[0], None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ))

fig = uplotly.lines2d(t111m[2], u111m[2,:,-1, 1]*0.01, fig,
                      dict(name="NASTRAN3",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
#fig.update_xaxes(range=[0, 4])
fig.update_xaxes(range=[0, 4], title='time [s]',tickfont = dict(size=16), titlefont=dict(size=16),
                                    mirror=True,
             ticks='outside',
             showline=True,
             linecolor='black',
             gridcolor='lightgrey')
fig.update_yaxes(title='$\hat{u}_y$', tickfont = dict(size=16),titlefont=dict(size=16),
                                    mirror=True,
           ticks='outside',
           showline=True,
           linecolor='black',
           gridcolor='lightgrey')
fig.update_layout(showlegend=False, plot_bgcolor='white',
    margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))

fig.show()

fig.write_image(f"../{figname}")
figname

name="Gust3Plot_z"

figname = f"figs/{name}.png"
x, y = putils.pickIntrinsic2D(sol_g3.data.dynamicsystem_s1.t,
                              sol_g3.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=150, dim=2))

fig = uplotly.lines2d(x[:], (y[:]-y[0])*gscale, None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ))

fig = uplotly.lines2d(t111m[2], u111m[2,:,-1, 2]*0.01*gscale, fig,
                      dict(name="NASTRAN",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
fig.update_xaxes(range=[0, 4], title='time [s]', titlefont=dict(size=16), tickfont = dict(size=16),
                                    mirror=True,
                 ticks='outside',
                 showline=True,
                 linecolor='black',
                 gridcolor='lightgrey')
fig.update_yaxes(title='$\hat{u}_z$', titlefont=dict(size=16), tickfont = dict(size=16),
                 mirror=True,
                 ticks='outside',
                 showline=True,
                 linecolor='black',
                 gridcolor='lightgrey')
fig.update_layout(plot_bgcolor='white',margin=dict(
      autoexpand=True, 
      l=0,
      r=0,
      t=0,
      b=0
  ))

fig.show()

fig.write_image(f"../{figname}")
figname

name="Gust4Plot_x"

figname = f"figs/{name}.png"
sol_g4 = solution.IntrinsicReader("./Gust4")
x, y = putils.pickIntrinsic2D(sol_g4.data.dynamicsystem_s1.t,
                              sol_g4.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=150, dim=0))

fig = uplotly.lines2d(x[1:], (y[:-1]-y[0])*gscale, None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ))

fig = uplotly.lines2d(t111m[2], u111m[2,:,-1, 0]*2*gscale, fig,
                      dict(name="NASTRAN",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
fig.update_xaxes(range=[0, 4], title='time [s]', titlefont=dict(size=16), tickfont = dict(size=16),
                                    mirror=True,
             ticks='outside',
             showline=True,
             linecolor='black',
             gridcolor='lightgrey')
fig.update_yaxes(title='$\hat{u}_x$', titlefont=dict(size=16), tickfont = dict(size=16),
                                    mirror=True,
           ticks='outside',
           showline=True,
           linecolor='black',
           gridcolor='lightgrey')
fig.update_layout(plot_bgcolor='white',showlegend=False,
                  margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))

fig.show()

fig.write_image(f"../{figname}")
figname

name="Gust4Plot_y"

figname = f"figs/{name}.png"
x, y = putils.pickIntrinsic2D(sol_g4.data.dynamicsystem_s1.t,
                              sol_g4.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=150, dim=1))

fig = uplotly.lines2d(x[1:], (y[:-1]-y[0])*gscale, None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ))

fig = uplotly.lines2d(t111m[2], u111m[2,:,-1, 1]*2*gscale, fig,
                      dict(name="NASTRAN",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
fig.update_xaxes(range=[0, 4], title='time [s]', titlefont=dict(size=16), tickfont = dict(size=16),
                                    mirror=True,
               ticks='outside',
               showline=True,
               linecolor='black',
               gridcolor='lightgrey')
fig.update_yaxes(title='$\hat{u}_y$', titlefont=dict(size=16), tickfont = dict(size=16),
                                    mirror=True,
             ticks='outside',
             showline=True,
             linecolor='black',
             gridcolor='lightgrey')
fig.update_layout(plot_bgcolor='white',showlegend=False,
                  margin=dict(
                      autoexpand=True,
                      l=0,
                      r=0,
                      t=0,
                      b=0
                  ))

fig.show()

fig.write_image(f"../{figname}")
figname

name="Gust4Plot_z"

figname = f"figs/{name}.png"
x, y = putils.pickIntrinsic2D(sol_g4.data.dynamicsystem_s1.t,
                              sol_g4.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=150, dim=2))

fig = uplotly.lines2d(x[1:], (y[:-1]-y[0])*gscale, None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ))

fig = uplotly.lines2d(t111m[2], u111m[2,:,-1, 2]*2*gscale, fig,
                      dict(name="NASTRAN",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
fig.update_xaxes(range=[0, 4], title='time [s]', titlefont=dict(size=16), tickfont = dict(size=16),
                                    mirror=True,
             ticks='outside',
             showline=True,
             linecolor='black',
             gridcolor='lightgrey')
fig.update_yaxes(title='$\hat{u}_z$', titlefont=dict(size=16), tickfont = dict(size=16),
                                    mirror=True,
           ticks='outside',
           showline=True,
           linecolor='black',
           gridcolor='lightgrey')
fig.update_layout(plot_bgcolor='white',
    margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))

fig.show()

fig.write_image(f"../{figname}")
figname

name=[]
import numpy as np
directory = feniax.PATH / "../Models/XRF1-2/Results_modes/"
nmodes = 70
#q = np.load("%s/q_%s.npy"%(directory, nmodes))
omega = np.load("%s/../Results_modes/Omega_%s.npy"%(directory, nmodes))
alpha1 = np.load("%s/../Results_modes/alpha1_%s.npy"%(directory, nmodes))
alpha2 = np.load("%s/../Results_modes/alpha2_%s.npy"%(directory, nmodes))
gamma1 = np.load("%s/../Results_modes/gamma1_%s.npy"%(directory, nmodes))
gamma2 = np.load("%s/../Results_modes/gamma2_%s.npy"%(directory, nmodes))

name="XRF1Plot_alpha1old"
figname = f"figs/{name}.png"
fig = px.imshow(np.abs(alpha1-np.eye(70)),
                labels=dict(color="Absolute values"),
                color_continuous_scale="Blues"
                )
fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(size=16)),margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))
fig.update_xaxes(title='Mode',
                 titlefont=dict(size=16),
                 tickfont = dict(size=16)
                 # mirror=True,
                 # ticks='outside',
                 # showline=True,
                 # linecolor='black',
                 # gridcolor='lightgrey'
                 )
fig.update_yaxes(title='Mode', tickfont = dict(size=16),
                 titlefont=dict(size=16)
                 # mirror=True,
                 # ticks='outside',
                 # showline=True,
                 # linecolor='black',
                 # gridcolor='lightgrey'
                 )
#fig.update_traces(colorbar_tickfont=dict(size=26))
fig.write_image(f"../{figname}")
fig.show()
figname

name="XRF1Plot_alpha1"

figname = f"figs/{name}.png"
sol_x1 = solution.IntrinsicReader("./Gust3")
fig = px.imshow(np.abs(sol_x1.data.couplings.alpha1-np.eye(70)),
                labels=dict(color="Absolute values"),
                color_continuous_scale="Blues"
                )
fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(size=16)),margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))
fig.update_xaxes(title='Mode',
                 titlefont=dict(size=16),
                 tickfont = dict(size=16)
                 # mirror=True,
                 # ticks='outside',
                 # showline=True,
                 # linecolor='black',
                 # gridcolor='lightgrey'
                 )
fig.update_yaxes(title='Mode', tickfont = dict(size=16),
                 titlefont=dict(size=16)
                 # mirror=True,
                 # ticks='outside',
                 # showline=True,
                 # linecolor='black',
                 # gridcolor='lightgrey'
                 )

fig.write_image(f"../{figname}")
fig.show()
figname

name="XRF1Plot_alpha2old"
figname = f"figs/{name}.png"
fig = px.imshow(np.abs(alpha2-np.eye(70)),
                labels=dict(color="Absolute values"),
                color_continuous_scale="Blues"
                )
fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(size=16)),margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))
fig.update_xaxes(title='Mode',
                 titlefont=dict(size=16),
                 tickfont = dict(size=16)
                 # mirror=True,
                 # ticks='outside',
                 # showline=True,
                 # linecolor='black',
                 # gridcolor='lightgrey'
                 )
fig.update_yaxes(title='Mode', tickfont = dict(size=16),
                 titlefont=dict(size=16)
                 # mirror=True,
                 # ticks='outside',
                 # showline=True,
                 # linecolor='black',
                 # gridcolor='lightgrey'
                 )


fig.write_image(f"../{figname}")
fig.show()
figname

name="XRF1Plot_alpha2"
#px.colors.named_colorscales()
figname = f"figs/{name}.png"
fig = px.imshow(np.abs(sol_x1.data.couplings.alpha2-np.eye(70)),
                labels=dict(color="Absolute values"),
                color_continuous_scale="Blues"
                )
fig.update_layout(coloraxis_colorbar=dict(tickfont=dict(size=16)),margin=dict(
      autoexpand=True,
      l=0,
      r=0,
      t=0,
      b=0
  ))
fig.update_xaxes(title='Mode',
                 titlefont=dict(size=16),
                 tickfont = dict(size=16)
                 # mirror=True,
                 # ticks='outside',
                 # showline=True,
                 # linecolor='black',
                 # gridcolor='lightgrey'
                 )
fig.update_yaxes(title='Mode', tickfont = dict(size=16),
                 titlefont=dict(size=16)
                 # mirror=True,
                 # ticks='outside',
                 # showline=True,
                 # linecolor='black',
                 # gridcolor='lightgrey'
                 )

fig.write_image(f"../{figname}")
fig.show()
figname
