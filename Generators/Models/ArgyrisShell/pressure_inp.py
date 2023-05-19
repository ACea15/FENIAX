import numpy as np
import os
from wing_inp import *

for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])

model='ArgyrisShell25'

sol='400'
static=1
dynamic=0
rotation=0
P=1000*np.asarray([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.96])
aset1=[1]
numLoads=len(P)
numForce=na
Fl=[]
for i in range(numLoads):
    fj=[]
    for j in range(numForce):
        if j==numForce-1:
            fj.append(P[i]*100/na/2)
        else:
            fj.append(P[i]*100/na)
    Fl.append(fj)        
#Fl=[[0.05e3],[0.1e3],[0.3e3],[0.43e3],[0.7e3],[1.04e3]]#,[1.5e3],[1.85e3],[2.1e3]]#,[2.65e3],[3.e3],[3.6e3],[5.0e3],[5.2e3],[7.e3],[8.4e3]]
#Fl=[[0.1e3],[0.43e3],[1.04e3],[1.85e3],[2.65e3],[3.6e3],[5.2e3]]
Fd=[[[0.,-1.,0.] for j in range(numForce)] for i in range(numLoads)]
gridF=[range(500001,500001+na) for i in range(numLoads)]

Ml = None
