from pyNastran.op2.op2 import OP2
import feniax.plotools.nastranvtk.bdfdef as bdfdef
import feniax.plotools.interpolation as interpolation
from feniax.preprocessor import solution
import feniax.plotools.grid as grid
from pyNastran.bdf.bdf import BDF
import pandas as pd
import feniax.plotools.reconstruction as rec
import feniax.preprocessor.configuration as configuration
import jax.numpy as jnp

results = "results4g"
config = configuration.Config.from_file(f"./{results}/config.yaml")
sol = solution.IntrinsicReader(f"./{results}")

tn = 250
tf = 250
ra_movie = jnp.zeros((tn+1, 3))
ra_movie = ra_movie.at[:, 0].set(-jnp.linspace(0,tf,tn+1))
time_movie = jnp.linspace(0,tf,tn+1)
r, u = rec.rbf_based_movie("./NASTRAN/XRF1-144trim.bdf",#"/media/acea/work/projects/FENIAX/examples/XRF1/NASTRAN/XRF1-146run.bdf",
                     config.fem.X,
                     4,
                     sol.data.staticsystem_s1.ra,
                     time_movie, ra_movie,
                     sol.data.staticsystem_s1.Cab,
                     sol.data.modes.C0ab,
                     vtkpath="./results2_5gVTKdyn/conf",
                     plot_timeinterval=1,
                     plot_ref=False,
                     tolerance=1e-2,
                     size_cards=16,
                     rbe3s_full=False)
