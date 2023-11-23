import numpy as np
import pickle
directory = "../../Models/XRF1-2/146-0_121-0Ma-3/"
nmodes = 70

q = np.load("%s/q_%s.npy"%(directory, nmodes))
omega = np.load("%s/../Results_modes/Omega_%s.npy"%(directory, nmodes))

alpha1 = np.load("%s/../Results_modes/alpha1_%s.npy"%(directory, nmodes))
alpha2 = np.load("%s/../Results_modes/alpha2_%s.npy"%(directory, nmodes))
gamma1 = np.load("%s/../Results_modes/gamma1_%s.npy"%(directory, nmodes))
gamma2 = np.load("%s/../Results_modes/gamma2_%s.npy"%(directory, nmodes))


with open ("%s/../Results_modes/Phil_%s"%(directory, nmodes), 'rb') as fp:
    [phi0l,phi1l,phi1ml,phi2l,mphi1l,cphi2xl]  = pickle.load(fp)

with open ("%s/../Results_modes/Phi_%s"%(directory, nmodes), 'rb') as fp:
    [phi0,phi1,phi1m,phi2,mphi1,cphi2x]  = pickle.load(fp)

with open ("%s/../Results_modes/Geometry"%(directory), 'rb') as fp:
    BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = pickle.load(fp)

with open ("%s/Sols_%s"%(directory, nmodes), 'rb') as fp:
    ra0,ra,Rab,strain,kappa  = pickle.load(fp)
