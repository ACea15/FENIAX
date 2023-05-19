import numpy as np
import os

sol='103'
pch=1
numloads=0
pc=150
R=50.
theta0=180*2*np.pi/360
dtheta=theta0/(pc)
dx=R*np.sin(dtheta)/np.cos(dtheta/2)

NumBeams=pc
Clamped=1
BeamsClamped=[0]
Node0=np.array([0.,0.,0.])
BeamConn=[0]+range(pc-1)

model='ArgyrisSemiCircle_150'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])   
femname='S_0.bdf'


L=[dx for i in range(NumBeams)]
N=[1+1]+[1 for i in range(NumBeams-1)]
#W=[1. for i in range(NumBeams)]
#H=[0.1 for i in range(NumBeams)]
#TH=[0.01 for i in range(NumBeams)]
E=[7.2e6 for i in range(NumBeams)]
J=[1e4 for i in range(NumBeams)]
Area=[1. for i in range(NumBeams)]
NU=[0.3 for i in range(NumBeams)]
I1=[0.5e4 for i in range(NumBeams)]
I2=[0.5 for i in range(NumBeams)]
Direc=np.asarray([[np.sin(dtheta*(i+1))-np.sin(dtheta*i),-np.cos(dtheta*(i+1))+np.cos(dtheta*i),0] for i in range(pc)])
PID=[1 for i in range(NumBeams)]
rho=[1. for i in range(NumBeams)]

numLoads=0
numForce=0
Fl=[[1500.],[2000.],[2500.],[3000.]]
Fd=[[[0.,0.,1.]] for i in range(numLoads)]
gridF=[[[-1,-1]] for i in range(numLoads)]

