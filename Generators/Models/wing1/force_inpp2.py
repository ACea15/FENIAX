import numpy as np
import os
from wing_inp2 import *
#import pyNastran
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])

model='wingA320_15'

sol='400'
static = 0
dynamic = 1
rotation=0
FORCE=0
# #Loads
# FORCE1=1
# numLoads=1
# numForce=1
# Fl=[[4e2]]
# Ml=None
# Fd=[[[0.,0.,1.]] for i in range(numLoads)]
# #Md=[[[0.,-1.,0.]] for i in range(numLoads)]
# gridF=[[500030] for i in range(numLoads)]
na = 30
P=np.asarray([10.])
aset1=[1]
length= np.sqrt(20.**2+29.8507**2)
numLoads=len(P)
numForce=na
Fl=[]
for i in range(numLoads):
    fj=[]
    for j in range(numForce):
        if j==numForce-1:
            fj.append(P[i]*length/na/2)
        else:
            fj.append(P[i]*length/na)
    Fl.append(fj)
#Fl=[[0.05e3],[0.1e3],[0.3e3],[0.43e3],[0.7e3],[1.04e3]]#,[1.5e3],[1.85e3],[2.1e3]]#,[2.65e3],[3.e3],[3.6e3],[5.0e3],[5.2e3],[7.e3],[8.4e3]]
#Fl=[[0.1e3],[0.43e3],[1.04e3],[1.85e3],[2.65e3],[3.6e3],[5.2e3]]
Fd=[[[0.,0.,1.] for j in range(numForce)] for i in range(numLoads)]
gridF=[range(500001,500001+na) for i in range(numLoads)]
Ml = None

Acceleration0 = 0
Velocity0 = 0
Displacement0 = 0

ug=range(na)
vg=range(na)
ag=range(na)

def fv(x):
  return 1.5*np.array(0.,1.,1.)

def fu(x):
  return 2*x

def fu(x):
  return 2*x

#ti=2000.
tableti=[[[0.,0.],[4.,1.],[4.,0.],[15.,0.]]]
ti_max = 9.
ti_n = 3500
