import numpy as np
import pickle
import scipy.linalg

directory = "/home/ac5015/programs/FEM4INAS/Models/ArgyrisFrame_20/Test/2DProblem/results_2-8-2023_11-12"
nmodes = 120

q = np.load("%s/q_%s.npy"%(directory, nmodes))
omega = np.load("%s/Results_modes/Omega_%s.npy"%(directory, nmodes))

alpha1 = np.load("%s/Results_modes/alpha1_%s.npy"%(directory, nmodes))
alpha2 = np.load("%s/Results_modes/alpha2_%s.npy"%(directory, nmodes))
gamma1 = np.load("%s/Results_modes/gamma1_%s.npy"%(directory, nmodes))
gamma2 = np.load("%s/Results_modes/gamma2_%s.npy"%(directory, nmodes))


with open ("%s/Results_modes/Phil_%s"%(directory, nmodes), 'rb') as fp:
    [phi0l,phi1l,phi1ml,phi2l,mphi1l,cphi2xl]  = pickle.load(fp)

with open ("%s/Results_modes/Phi_%s"%(directory, nmodes), 'rb') as fp:
    [phi0,phi1,phi1m,phi2,mphi1,cphi2x]  = pickle.load(fp)

with open ("%s/Results_modes/Geometry"%(directory), 'rb') as fp:
    BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = pickle.load(fp)

with open ("%s/Sols_%s"%(directory, nmodes), 'rb') as fp:
    ra0,ra,Rab,strain,kappa  = pickle.load(fp)

Ka = np.load("../../Models/ArgyrisFrame_20/FEM/Kaa.npy")
Ma = np.load("../../Models/ArgyrisFrame_20/FEM/Maa.npy")
w, v = scipy.linalg.eigh(Ka, Ma)

Ka2 = np.load("./FEM/Ka.npy")
Ma2 = np.load("./FEM/Ma.npy")
w2, v2 = scipy.linalg.eigh(Ka2, Ma2)

save_eigs = False
if save_eigs:
    np.save("../ArgyrisFrame/FEM/eigenvals.npy", w2)
    np.save("../ArgyrisFrame/FEM/eigenvecs.npy", v2)


directory2 = "/home/ac5015/programs/FEM4INAS/Models/ArgyrisFrame_20/Test/3DProblem/results_2-8-2023_11-16/"
nmodes = 120

q3 = np.load("%s/q_%s.npy"%(directory2, nmodes))
# omega3 = np.load("%s/Results_modes/Omega_%s.npy"%(directory2, nmodes))

# alpha13 = np.load("%s/Results_modes/alpha1_%s.npy"%(directory2, nmodes))
# alpha23 = np.load("%s/Results_modes/alpha2_%s.npy"%(directory2, nmodes))
# gamma13 = np.load("%s/Results_modes/gamma1_%s.npy"%(directory2, nmodes))
# gamma23 = np.load("%s/Results_modes/gamma2_%s.npy"%(directory2, nmodes))

# with open ("%s/Results_modes/Phil_%s"%(directory2, nmodes), 'rb') as fp:
#     [phi0l3,phi1l3,phi1ml3,phi2l3,mphi1l3,cphi2xl3]  = pickle.load(fp)

# with open ("%s/Results_modes/Phi_%s"%(directory2, nmodes), 'rb') as fp:
#     [phi03,phi13,phi1m3,phi23,mphi13,cphi2x3]  = pickle.load(fp)

# with open ("%s/Results_modes/Geometry"%(directory2), 'rb') as fp:
#     BeamSeg3, NumNode3, NumNodes3, DupNodes3, inverseconn3  = pickle.load(fp)

with open ("%s/Sols_%s"%(directory2, nmodes), 'rb') as fp:
    ra03,ra3,Rab3,strain3,kappa3  = pickle.load(fp)
