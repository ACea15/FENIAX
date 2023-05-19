import pickle
import numpy as np

with open('argyris.pickle', 'rb') as handle:
    argy = pickle.load(handle)

    
disx=['0p05','0p1','0p2','0p3','0p4','0p5','0p6','0p7','0p8','0p9','0p96']
inid='beampre'

ini = ['beampre','c','cirVer','cirTran','frame']
dis = [['0p05','0p1','0p2','0p3','0p4','0p5','0p6','0p7','0p8','0p9','0p96'],
       ['3p7','7p6','12p1','15p5','17p5','25p2','39p3','48p2','61','80','94p5','109p5','120'],
       ['0p2','0p4','0p6','0p8','1p0','1p2','1p4','1p6','2p0'],
       ['0','0p2','0p6','1p2','1p8','2p2','2p6','3p2','3p6','4p2'],
       ['0p1','0p43','1p04','1p85','2p65','3p6','5p0','5p2','8p4']]

argynew = {}
def trans(disx,inid):
    discrete=[]
    ar=[]
    for i in range(len(disx)):
       discrete.append([inid+disx[i]+'x',inid+disx[i]+'y'])
       ar.append([argy[discrete[i][0]],argy[discrete[i][1]]])
    argynew[inid]=ar

for i in range(len(ini)):
    trans(dis[i],ini[i])

with open('argyris_new.pickle', 'wb') as fp:
    pickle.dump(argynew, fp)
 
# from scipy.interpolate import interp1d
# u1=interp1d(argy['su1x'],argy['su1y'])
# u2=interp1d(argy['su2x'],argy['su2y'])
# u3=interp1d(argy['su3x'],argy['su3y'])
