import numpy as np
import os

sol='103'
pch=1
numloads=4
pc=50
R=500.
theta0=180*2*np.pi/360
dtheta=theta0/(pc)
dx=R*np.sin(dtheta)/np.cos(dtheta/2)

NumBeams=pc
Clamped=1
BeamsClamped=[0]
Node0=np.array([0.,0.,0.])
BeamConn=[0]+range(pc-1)


model='nallazambi_circle50'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])   
femname='S_0.bdf'


L=[dx for i in range(NumBeams)]
N=[2]+[1 for i in range(NumBeams-1)]
W=[1. for i in range(NumBeams)]
H=[1. for i in range(NumBeams)]
TH=[0.1 for i in range(NumBeams)]
E=[72000. for i in range(NumBeams)]
J=[72000e4 for i in range(NumBeams)]
Area=[2000. for i in range(NumBeams)]
NU=[0. for i in range(NumBeams)]
I1=[5000. for i in range(NumBeams)]
I2=[5000. for i in range(NumBeams)]
Direc=np.asarray([[np.sin(dtheta*(i+1))-np.sin(dtheta*i),np.cos(dtheta*(i+1))-np.cos(dtheta*i),0] for i in range(pc)])
PID=[1 for i in range(NumBeams)]
rho=[2.703e3 for i in range(NumBeams)]

numLoads=0
numForce=1
Fl=[[1500.],[2000.],[2500.],[3000.]]
Fd=[[[0.,0.,1.]] for i in range(numLoads)]
gridF=[[[-1,-1]] for i in range(numLoads)]

