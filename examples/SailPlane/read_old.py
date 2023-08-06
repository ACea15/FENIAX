import numpy as np
import pickle
import scipy.linalg

alpha1 = np.load("/home/ac5015/programs/FEM4INAS/Models/SailPlane/Test/Static/results_5-6-2023_13-37/Results_modes/alpha1_50.npy")

alpha2 = np.load("/home/ac5015/programs/FEM4INAS/Models/SailPlane/Test/Static/results_5-6-2023_13-37/Results_modes/alpha2_50.npy")

with open ("/home/ac5015/programs/FEM4INAS/Models/SailPlane/Test/Static/results_5-6-2023_13-37/Results_modes/Phil_50", 'rb') as fp:
    [phi0l,phi1l,phi1ml,phi2l,mphi1l,cphi2xl]  = pickle.load(fp)

with open ("/home/ac5015/programs/FEM4INAS/Models/SailPlane/Test/Static/results_5-6-2023_13-37/Results_modes/Phi_50", 'rb') as fp:
    [phi0,phi1,phi1m,phi2,mphi1,cphi2x]  = pickle.load(fp)

with open ("/home/ac5015/programs/FEM4INAS/Models/SailPlane/Test/Static/results_5-6-2023_13-37/Results_modes/Geometry", 'rb') as fp:
    BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = pickle.load(fp)

    
omega = np.load("/home/ac5015/programs/FEM4INAS/Models/SailPlane/Test/Static/results_5-6-2023_13-37/Results_modes/Omega_50.npy")

# Ka = np.load("../../Models/ArgyrisFrame_20/FEM/Kaa.npy")
# Ma = np.load("../../Models/ArgyrisFrame_20/FEM/Maa.npy")
# w, v = scipy.linalg.eigh(Ka, Ma)

Ka2 = np.load("./FEM/Ka.npy")
Ma2 = np.load("./FEM/Ma.npy")
w2, v2 = scipy.linalg.eigh(Ka2, Ma2)
np.save("./FEM/w.npy", w2)
np.save("./FEM/v.npy", v2)

