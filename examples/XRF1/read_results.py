from pyNastran.op2.op2 import OP2
import pandas as pd
import plotly.express as px
import fem4inas.preprocessor.solution as solution



# op2 = OP2()
# op2.set_additional_matrices_to_read({b'OPHP':False})
# op2.read_op2("./NASTRAN/SOL146/run_cao.op2")



# data = op2.displacements[1].data
# df = pd.DataFrame(dict(#time=op2.displacements[1].dts,
#                   z=data[:,19,2]))

# fig = px.line(df,  title='Life expectancy in Canada')
#fig.show()

import pickle
import numpy as np
directory = "../../Models/XRF1-2/146-0_121-0Ma-3/"
nmodes = 70

q = np.load("%s/q_%s.npy"%(directory, nmodes))
omega = np.load("%s/../Results_modes/Omega_%s.npy"%(directory, nmodes))

alpha1 = np.load("%s/../Results_modes/alpha1_%s.npy"%(directory, nmodes))
alpha2 = np.load("%s/../Results_modes/alpha2_%s.npy"%(directory, nmodes))
gamma1 = np.load("%s/../Results_modes/gamma1_%s.npy"%(directory, nmodes))
gamma2 = np.load("%s/../Results_modes/gamma2_%s.npy"%(directory, nmodes))
# with open ("oldResults/ra", 'r') as fp:
#     ra  = pickle.load(fp)
ra  = np.load("oldResults/ra.npy")


gust = solution.IntrinsicReader("resultsGust")

df = pd.DataFrame(dict(#time=op2.displacements[1].dts,
    z=gust.data.dynamicsystem_s1.ra[:,2, 41],
    zold=ra[:,2],
    z0=gust.data.dynamicsystem_s1.ra[0,2, 41]*np.ones(3001),
    z02= ra[0,2]*np.ones(3001)
                  ))

fig = px.line(df,  title='')
fig.show()



fig = px.imshow(alpha2)
fig.show()

fig = px.imshow(np.abs(alpha2-np.eye(70)))
fig.show()

fig = px.imshow(np.abs(gust.data.couplings.alpha2-np.eye(70)))
fig.show()
