# [[file:modelgeneration.org::*Read op4][Read op4:1]]
import numpy as np
import feniax.unastran.op4handler as op4handler

Qax_name = "QaxDd1c7F3Seao-100"
Qah_name = "QahDd1c7F3Seao-100"
Qhx_name = "QhxDd1c7F3Seao-100"
Qax = op4handler.read_data(f'./NASTRAN/data_out/{Qax_name}.op4',
                           'Q_AX')
Qah = op4handler.read_data(f'./NASTRAN/data_out/{Qah_name}.op4',
                           'Q_AH')
Qhx = op4handler.read_data(f'./NASTRAN/data_out/{Qhx_name}.op4',
                           'Q_HX')
SAVE_Qx = True
if SAVE_Qx:
    np.save(f"./AERO/{Qax_name}.npy", Qax)
    np.save(f"./AERO/{Qah_name}.npy", Qah)
    np.save(f"./AERO/{Qhx_name}.npy", Qhx)
# Read op4:1 ends here
