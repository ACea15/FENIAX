output="num_modes"
import numpy as np
import fem4inas.unastran.aero as nasaero
mach = 0.8
Mach = str(mach).replace('.','_')
machs = [mach]
reduced_freqs = np.hstack([1e-6, np.linspace(1e-3,1, 50)]) #np.hstack([np.linspace(1e-5,1, 50), [10-0.001, 10., 10+0.001]])
num_modes = 50
chord_panels = dict(wing=15, hstabilizer=10, vstabilizer=10)
#aero['s_ref'] = 361.6
#aero['b_ref'] = 58.0
#aero['X_ref'] = 36.3495
flutter_id = 9010
mach_fact = machs
kv_fact = [200., 220.]
u_inf = 200.
rho_inf = 1.5
density_fact = [rho_inf]
c_ref = 1.
b_ref = 28.8*2
S_ref = b_ref * c_ref
rho_ref=rho_inf
q_inf = 0.5 * rho_inf * u_inf ** 2
alpha_sol144 = 1 * np.pi / 180
flutter_method="PK"
flutter_sett = dict()
aero_sett = dict()
num_poles = 5
gust_lengths = [18.0,42.,67.,91.,116.,140.,165.0,189.,214.]
eval(output)

#from fem4inas.utils import write_op4modes
import fem4inas.unastran.op4handler as op4handler
eigs, modes = op4handler.write_op4modes("./run_caof",
                             num_modes,
                             op4_name=f"./data_out/Phi{num_modes}",
                             return_modes=True)

# num_nodes = 79
# eigs = np.array(eigs)
# modes4sims = np.zeros((6*(num_nodes - 1), num_modes))
# for i in range(num_modes):
#     modes4sims[:, i] = np.hstack(modes[i,:(num_nodes - 1)])
# SAVE = True
# #np.load("../FEM/Ka.npy")
# #scipy.linalg.eigh(Ka, Ma)
# if SAVE:
#     np.save("../FEM/eigenvecs.npy", modes4sims)
#     np.save("../FEM/eigenvals.npy", eigs)

dlm_gafs = nasaero.GenFlutter(flutter_id,
                              density_fact,
                              mach_fact,
                              kv_fact,
                              machs,
                              reduced_freqs,
                              u_inf,
                              c_ref,
                              rho_ref,
                              flutter_method,
                              flutter_sett,
                              aero_sett)

dlm_gafs.build_model()
dlm_gafs.model.write_bdf("./GAFs/aero_flutter.bdf")

import pyNastran.op4.op4 as op4
from scipy.io import savemat

READ_Qhh = False
if READ_Qhh:
    Qhh = op4.read_op4(f"./data_out/Qhh{Mach}-{num_modes}.op4")
    # savemat(f"./GAFs/matlab_Qhh{Mach}-{num_modes}.mat", dict(Qhh=Qhh['Q_HH'][1],
    #                                                          reduced_freqs=reduced_freqs))
