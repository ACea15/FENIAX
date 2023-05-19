import numpy as np
import os
#import pyNastran

model='Shell_Hesse50'

sol='103'
numloads=1
pch=1
dynamic = 0
static = 0
free =1
# Geometry
#=============================================================
Lx=10.; Lyr=np.sqrt(0.3) ; Lyt=np.sqrt(0.3) ; Lzr = np.sqrt(0.3) ; Lzt = np.sqrt(0.3)
nx=50; ny=5; nz=5

dlx=Lx/(nx-1); dly=Lyr/(ny-1); dlz=Lzr/(nz-1)
tipy=0; tipz=0

# Shell  properties
# =============================================================
Em=5./6*1E+5
Nu=0.
thickness=np.sqrt(0.3)/10

# Condensation points
na=50
#aset=np.linspace(dlx,Lx,na)
aset=np.linspace(0,Lx,na)

#Mass
rho = 10./3
lumped = 1
Area=0.3
I2=10.*Lx/(na)*np.ones(na)
I1=10.*Lx/na*np.ones(na)
Mass=rho*Area*Lx/(na)*np.ones(na)

I=np.zeros((6,na))
for i in range(na):
  I[0,i]=I1[i]+I2[i];I[1,i]=0.;I[2,i]=I1[i];I[3,i]=0.0;I[4,i]=0.0;I[5,i]=I2[i]
 #I11        I21       I22        I31        I32        I33


# Rotate model
rot_angle = np.arctan(-8./6)
rot_direc = np.array([0.,0.,1.])
rotation=1



#Loads
FORCE1=0
numLoads=1
numForce=1
Fl=[[50.],[100.],[200.],[300.],[450.],[550.]]
Ml=None
Fd=[[[0.,0.,-1.]] for i in range(numLoads)]
Md=[[[0.,-1.,0.]] for i in range(numLoads)]
gridF=[[-1] for i in range(numLoads)]

Acceleration0 = 0
Velocity0 = 1
Displacement0 = 0

ug=range(na)
vg=range(na)
ag=range(na)

def fv(x):
  return 0.05*np.array([0.,x[0],x[0]])/20

def fu(x):
  return 1.5*np.array([0.,1.,1.])

def fa(x):
  return 2*x

tni=5000
tf=20.


for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])

femName = 'S_na%s.bdf' %(na)
