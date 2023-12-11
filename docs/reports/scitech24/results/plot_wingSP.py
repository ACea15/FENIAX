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

wingSP_folder = fem4inas.PATH / "../examples/wingSP"
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
