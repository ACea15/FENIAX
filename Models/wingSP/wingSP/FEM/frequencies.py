import  numpy as np
import scipy.linalg

Ka=np.load('Ka.npy')
Ma=np.load('Ma.npy')
DKa,VKa =  scipy.linalg.eigh(Ka,Ma)

Ki=np.load('Ki.npy')
Mi=np.load('Mi.npy')
DKi,VKi =  scipy.linalg.eigh(Ki,Mi)
######################################

# K_ribhole=np.load('K_ribhole.npy')
# M_ribhole=np.load('M_ribhole.npy')
# DK_ribhole,VK_ribhole =  scipy.linalg.eigh(K_ribhole,M_ribhole)


Ka_ribhole=np.load('Ka_ribhole.npy')
Ma_ribhole=np.load('Ma_ribhole.npy')
DKa_ribhole,VKa_ribhole =  scipy.linalg.eigh(Ka_ribhole,Ma_ribhole)

Ki_ribhole=np.load('Ki_ribhole.npy')
Mi_ribhole=np.load('Mi_ribhole.npy')
DKi_ribhole,VKi_ribhole =  scipy.linalg.eigh(Ki_ribhole,Mi_ribhole)

######################################
# K_norib=np.load('K_norib.npy')
# M_norib=np.load('M_norib.npy')
# DK_norib,VK_norib =  scipy.linalg.eigh(K_norib,M_norib)


Ka_norib=np.load('Ka_norib.npy')
Ma_norib=np.load('Ma_norib.npy')
DKa_norib,VKa_norib =  scipy.linalg.eigh(Ka_norib,Ma_norib)

Ki_norib=np.load('Ki_norib.npy')
Mi_norib=np.load('Mi_norib.npy')
DKi_norib,VKi_norib =  scipy.linalg.eigh(Ki_norib,Mi_norib)

######################################
