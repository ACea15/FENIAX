# [[file:modelgeneration.org::*Read op4][Read op4:1]]
import numpy as np
import feniax.unastran.op4handler as op4handler
dihedral = op4handler.read_data(f'./NASTRAN/data_out/Dihedral.op4',
                           'WJ')
SAVE_DIHEDRAL = True
if SAVE_DIHEDRAL:
    np.save(f"./AERO/Dihedral_{label_dlm}.npy", dihedral.real[:,0])
# Read op4:1 ends here
