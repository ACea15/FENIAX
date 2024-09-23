import numpy as np
from pyNastran.op4.op4 import read_op4

def frequency_matrix(k_array, poles):
  num_reducedfreq = len(k_array)
  num_poles = len(poles)
  k_array2 = k_array ** 2
  k_matrix = np.zeros((num_reducedfreq, 3 + num_poles), dtype=complex)
  k_matrix[:,0]=1.0
  k_matrix[:,1]=k_array*1j
  k_matrix[:,2]=-k_array2
  for i, pi in enumerate(poles):
    k_matrix[:,3+i]=1j*k_array/(pi+1j*k_array)
  return k_matrix

def Q_RFA(k_array, roger_matrices, poles):
  k_matrix_comp = frequency_matrix(k_array, poles)
  term1=np.einsum('ij,jkl->ikl', k_matrix_comp, roger_matrices[1:])
  return term1+roger_matrices[0]

def rogerRFA(k_matrix_comp, Qk):
  lhs=Qk[1:]
  lhs_expand=np.concatenate((lhs.real,lhs.imag),axis=0)
  k_expand=np.concatenate((k_matrix_comp.real,k_matrix_comp.imag),axis=0)
  inv_k=np.linalg.pinv(k_expand)
  rhs_expand=np.einsum('ij,jkl->ikl', inv_k, lhs_expand)
  return rhs_expand

def process_Q(op4name,poles=None):
  if poles==None:
    poles=np.linspace(1e-3, 5, 20)
  aero=read_op4(op4name)
  key=list(aero.keys())[0]
  q=np.array(aero[key].data)
  k_array=np.linspace(0,1,50)
  k_matrix=frequency_matrix(k_array, poles)
  roger_matrices=rogerRFA(k_matrix,q)
  return roger_matrices