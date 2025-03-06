import numpy as np
import pickle
import scipy.linalg

directory = "/home/ac5015/programs/FEM4INAS/Models/Simo45_15/Test/FollowerForce/results_5-6-2023_13-46/"
#directory = "/media/acea/work/projects/FEM4INAS/Models/Simo45_15/Test/FollowerForce/results_26-4-2023_7-20/"
nmodes = 90

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


directory = "/home/ac5015/programs/FEM4INAS/Models/Simo45_15/Test/DeadForce/results_5-6-2023_13-38"
nmodes = 90

qd = np.load("%s/q_%s.npy"%(directory, nmodes))
omegad = np.load("%s/Results_modes/Omega_%s.npy"%(directory, nmodes))

alpha1d = np.load("%s/Results_modes/alpha1_%s.npy"%(directory, nmodes))
alpha2d = np.load("%s/Results_modes/alpha2_%s.npy"%(directory, nmodes))
gamma1d = np.load("%s/Results_modes/gamma1_%s.npy"%(directory, nmodes))
gamma2d = np.load("%s/Results_modes/gamma2_%s.npy"%(directory, nmodes))


with open ("%s/Results_modes/Phil_%s"%(directory, nmodes), 'rb') as fp:
    [phi0ld,phi1ld,phi1mld,phi2ld,mphi1ld,cphi2xld]  = pickle.load(fp)

with open ("%s/Results_modes/Phi_%s"%(directory, nmodes), 'rb') as fp:
    [phi0d,phi1d,phi1md,phi2d,mphi1d,cphi2xd]  = pickle.load(fp)

with open ("%s/Results_modes/Geometry"%(directory), 'rb') as fp:
    BeamSegd, NumNoded, NumNodesd, DupNodesd, inverseconnd  = pickle.load(fp)

with open ("%s/Sols_%s"%(directory, nmodes), 'rb') as fp:
    ra0d,rad,Rabd,straind,kappad  = pickle.load(fp)




    
Ka = np.load("../../Models/Simo45_15/FEM/Kaa.npy")
Ma = np.load("../../Models/Simo45_15/FEM/Maa.npy")
w, v = scipy.linalg.eigh(Ka, Ma)

Ka2 = np.load("./FEM/Ka.npy")
Ma2 = np.load("./FEM/Ma.npy")
w2, v2 = scipy.linalg.eigh(Ka2, Ma2)

save_eigs = False
if save_eigs:
    np.save("../Simo45Beam/FEM/Ka.npy", Ka)
    np.save("../Simo45Beam/FEM/Ma.npy", Ma)
    
    np.save("../Simo45Beam/FEM/eigenvals.npy", w)
    np.save("../Simo45Beam/FEM/eigenvecs.npy", v)
