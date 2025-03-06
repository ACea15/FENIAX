import numpy as np
import feniax.unastran.aero as nasaero
import feniax.unastran.op4handler as op4handler
import pickle
import importlib
import feniax.aeromodal.roger as roger
importlib.reload(roger)
#op4m = op4.OP4()
#Qop4 = op4m.read_op4(file_name)

Qhh = op4handler.read_data('./NASTRAN/data_out/QhhDd1c7F3Seao-100.op4',
                           'Q_HH')
Qhj = op4handler.read_data('./NASTRAN/data_out/QhjDd1c7F3Seao-100.op4',
                           'Q_HJ')

num_poles = 5
Dhj_file = f"D{label_gaf}p{num_poles}"
Ahh_file = f"A{label_gaf}p{num_poles}"
Poles_file = f"Poles{label_gaf}p{num_poles}"    
optpoles = roger.OptimisePoles(reduced_freqs, Qhh,
                               num_poles_=num_poles,
                               poles_step_=0.3,
                               poles_range_=[0.05,3],
                               rfa_method_=1
                               )
# optpoles.set_errsettings(#error_name="max",
#                          rfa_method=2,
#                          norm_order=None)
optpoles.run(show_info=True)

qhhr1 = optpoles.get_model(label='m1')
# optimize with method 2
optpoles.set_errsettings(error_name="average", rfa_method=2, norm_order=None)
optpoles.run(show_info=True)
qhhr2 = optpoles.get_model(label='m2')
# optimize for max function
optpoles.set_errsettings(error_name="max", rfa_method=2, norm_order=None)
optpoles.run(show_info=True)
optpoles.save("./AERO", Ahh_file,
              Poles_file)
qhhr3 = optpoles.get_model(label='m3')
Qroger1 = qhhr1.eval_array(reduced_freqs)
Qroger2 = qhhr2.eval_array(reduced_freqs)
Qroger3 = qhhr3.eval_array(reduced_freqs)

poles = qhhr3.poles #jnp.load("./AERO/PolesDd1c7F1Scao-50.npy")
rogerhj = roger.ComputeRoger(Qhj, reduced_freqs, poles, 2)
np.save(f"./AERO/{Dhj_file}.npy", rogerhj.roger_matrices)
rogerhjeval = roger.EvaluateRoger.create(rogerhj)
Qrogerhj = rogerhjeval.eval_array(reduced_freqs)
