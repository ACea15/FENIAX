import numpy as np
import Utils.common

Kaa = np.load('Kaa.npy')
Maa = np.load('Maa.npy')

Ka,dx,dy = Utils.common.remove_zeros(Kaa,1)
Ma,dxm,dym = Utils.common.remove_zeros(Maa,1)

np.save('Ka.npy',Ka)
np.save('Ma.npy',Ma)
