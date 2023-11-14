import sys
import os
import pdb
# sys.path.append(os.getcwd())
import numpy as np
from variables import *

Grid = 'structuralGrid'
K_a = 'Ka.npy'
M_a = 'Ma.npy'
#M_a2 = 'Maa.npy'
#K_a2 = 'Kaa.npy'

Grid = feminas_dir + model_name + '/FEM/' + Grid
K_a = feminas_dir + model_name + '/FEM/' + K_a
M_a = feminas_dir + model_name + '/FEM/' + M_a
#M_a2 = feminas_dir + model_name + '/FEM/' + M_a2
#K_a2 = feminas_dir + model_name + '/FEM/' + K_a2
Ka=np.load(K_a)
Ma=np.load(M_a)
#Ka2=np.load(K_aa)
#Ma2=np.load(M_aa)
import scipy.linalg
w,v=scipy.linalg.eigh(Ka,Ma)
ww=np.sqrt(w)

if (__name__ == '__main__'):


  saving=1
  if saving:
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_config
    #pdb.set_trace()
    write_config(locals())

  print('Running Variables')
