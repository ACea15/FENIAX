# [[file:modelgeneration.org::parameters_gafs0][parameters_gafs0]]
import numpy as np
import json
import feniax.unastran.aero as nasaero
import feniax.unastran.op4handler as op4handler
from feniax.utils import standard_atmosphere
import pickle
import itertools
sol = "eao"
num_modes = 100
mach = 0.7
altitude = 10000 # meters
Mach = str(mach).replace('.','_')
machs = [mach]
reduced_freqs = np.hstack([1e-6, np.linspace(1e-5,1e-1, 25),
                           np.linspace(1e-1,5e-1, 25)[1:],
                           np.linspace(5e-1, 1., 10)[1:]])
reduced_freqs = np.hstack([1e-5, np.linspace(1e-4, 1, 100)
                           ])
#reduced_freqs = np.geomspace(1e-5, 1, 100, endpoint=True)
flutter_id = 9010
mach_fact = machs
kv_fact = [200., 220.]
T, rho_inf, P, a = standard_atmosphere(altitude)
u_inf = mach * a
#rho_inf = 1.2
density_fact = [rho_inf]
chord_ref = 3.
span_ref = 24. * 2  # always full span
area_ref = span_ref * chord_ref # make it half full area if half model
rho_ref=rho_inf
q_inf = 0.5 * rho_inf * u_inf ** 2
flutter_method="PK"
flutter_sett = dict()
aero_sett = dict()
label_dlm = "d1c7"
label_flow = f"F3"
label_gaf = f"D{label_dlm}{label_flow}S{sol}-{num_modes}"
input_dict = dict(reduced_freqs=list(reduced_freqs), mach=mach, u_inf=u_inf, rho_inf=rho_inf)
with open(f"./NASTRAN/GAFs/input_{label_flow}.json", "w") as fp:
    json.dump(input_dict, fp)  # encode dict into JSON
# parameters_gafs0 ends here

# [[file:modelgeneration.org::*Unsteady][Unsteady:1]]
dlm_gafs = nasaero.GenFlutter(flutter_id,
                              density_fact,
                              mach_fact,
                              kv_fact,
                              machs,
                              reduced_freqs,
                              u_inf,
                              chord_ref,
                              rho_ref,
                              flutter_method,
                              flutter_sett,
                              aero_sett)

dlm_gafs.build_model()
dlm_gafs.model.write_bdf(f"./NASTRAN/GAFs/{label_flow}.bdf")
# Unsteady:1 ends here
