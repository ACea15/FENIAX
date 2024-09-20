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

wingSP_folder = feniax.PATH / "../examples/wingSP"
nastran_path = wingSP_folder / "NASTRAN/"
nas_wspl = op2reader.NastranReader(op2name=(nastran_path / "wing_109d.op2"),
                                   bdfname=(nastran_path / "wing_109b.bdf"))
nas_wspl.readModel()
t_wspl, u_wspl = nas_wspl.displacements()  
###
nas_wsp = op2reader.NastranReader(op2name=(nastran_path / "wing400d.op2"),
                                   bdfname=(nastran_path / "wing_109b.bdf"))
nas_wsp.readModel()
t_wsp, u_wsp = nas_wsp.displacements()

name="wingSP_z"
figname = f"figs/{name}.png"
sol_wsp1 = solution.IntrinsicReader("./wingSP")
x, y = putils.pickIntrinsic2D(sol_wsp1.data.dynamicsystem_s1.t,
                              sol_wsp1.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=23, dim=2))

fig = uplotly.lines2d(x, y - y[0], None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ),
                      dict())
fig = uplotly.lines2d(t_wsp[0], u_wsp[0,:,-4, 2], fig,
                      dict(name="NASTRAN",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
#fig.update_xaxes(range=[0, 5])
fig.write_image(f"../{figname}")
fig.show()
figname

name="wingSP_x"
figname = f"figs/{name}.png"
sol_wsp1 = solution.IntrinsicReader("./wingSP")
x, y = putils.pickIntrinsic2D(sol_wsp1.data.dynamicsystem_s1.t,
                              sol_wsp1.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=23, dim=0))

fig = uplotly.lines2d(x, y - y[0], None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ),
                      dict())
fig = uplotly.lines2d(t_wsp[0], u_wsp[0,:,-4, 0], fig,
                      dict(name="NASTRAN",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
#fig.update_xaxes(range=[0, 5])
fig.write_image(f"../{figname}")
fig.show()
figname

name="wingSP_y"
figname = f"figs/{name}.png"
sol_wsp1 = solution.IntrinsicReader("./wingSP")
x, y = putils.pickIntrinsic2D(sol_wsp1.data.dynamicsystem_s1.t,
                              sol_wsp1.data.dynamicsystem_s1.ra,
                              fixaxis2=dict(node=23, dim=1))

fig = uplotly.lines2d(x, y - y[0], None,
                      dict(name="NMROM",
                           line=dict(color="navy")
                           ),
                      dict())
fig = uplotly.lines2d(t_wsp[0], u_wsp[0,:,-4, 1], fig,
                      dict(name="NASTRAN",
                           line=dict(color="grey",
                                     dash="dash")
                           ))
#fig.update_xaxes(range=[0, 5])
fig.write_image(f"../{figname}")
fig.show()
figname
