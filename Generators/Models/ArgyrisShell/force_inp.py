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

aset1=[1]
numLoads=6
numForce=1
Fl=[[3.75e3],[7.6e3],[12.1e3],[15.5e3],[17.5e3],[25.2e3]]#,[1.5e3],[1.85e3],[2.1e3]]#,[2.65e3],[3.e3],[3.6e3],[5.0e3],[5.2e3],[7.e3],[8.4e3]]
#Fl=[[0.1e3],[0.43e3],[1.04e3],[1.85e3],[2.65e3],[3.6e3],[5.2e3]]
Fd=[[[0.,-1.,0.] for j in range(numForce)] for i in range(numLoads)]
gridF=[[500025] for i in range(numLoads)]

Ml = None
