import numpy as np
import pickle
import scipy.linalg

alpha1 = np.load("/media/acea/work/projects/FEM4INAS/Models/ArgyrisBeam_25\
/Test/EndLoad/results_29-7-2023_18-18/Results_modes/alpha1_150.npy")

alpha2 = np.load("/media/acea/work/projects/FEM4INAS/Models/ArgyrisBeam_25\
/Test/EndLoad/results_29-7-2023_18-18/Results_modes/alpha2_150.npy")

with open ("/media/acea/work/projects/FEM4INAS/Models/ArgyrisBeam_25/Test/EndLoad/results_29-7-2023_18-18/Results_modes/Phil_150", 'rb') as fp:
    [phi0l,phi1l,phi1ml,phi2l,mphi1l,cphi2xl]  = pickle.load(fp)

with open ("/media/acea/work/projects/FEM4INAS/Models/ArgyrisBeam_25/Test/EndLoad/results_29-7-2023_18-18/Results_modes/Phi_150", 'rb') as fp:
    [phi0,phi1,phi1m,phi2,mphi1,cphi2x]  = pickle.load(fp)


Ka = np.load("/media/acea/work/projects/FEM4INAS/Models/ArgyrisBeam_25/FEM/Kaa.npy")
Ma = np.load("/media/acea/work/projects/FEM4INAS/Models/ArgyrisBeam_25/FEM/Maa.npy")
w, v = scipy.linalg.eigh(Ka, Ma)
