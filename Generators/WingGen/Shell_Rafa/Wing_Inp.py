import numpy as np
import os
#import pyNastran

model='Shell_Rafa30_rho'

sol='400'
numloads=1
pch=0
dynamic = 1
static = 0
free = 0
rotation = 0
# Geometry
#=============================================================
Lx=20.; Lyr=1. ; Lyt=1. ; Lzr = 0.1 ; Lzt=0.1
nx=31; ny=5; nz=5

dlx=Lx/(nx-1); dly=Lyr/(ny-1); dlz=Lzr/(nz-1)
tipy=0; tipz=0

# Shell  properties
# =============================================================
Em=1.E+6
Nu=0.3
thickness=0.01

# Condensation points
na=30
aset=np.linspace(dlx,Lx,na)

#Mass
rho = 1.
lumped = 0
Area=2*thickness*Lyr+2*thickness*Lzr
I2=rho*Lx/(na+1)*(2*Lyr*thickness*(Lzr/2)**2+2*(1./12)*thickness*Lzr**3)
I1=rho*Lx/(na+1)*(2*Lzr*thickness*(Lyr/2)**2+2*(1./12)*thickness*Lyr**3)
Mass=rho*Area*Lx/(na+1)*np.ones(na)

I=np.zeros((6,na))
for i in range(na):
  I[0,i]=I1+I2;I[1,i]=0.;I[2,i]=I1;I[3,i]=0.0;I[4,i]=0.0;I[5,i]=I2
 #I11        I21       I22        I31        I32        I33


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

ti=5000
lamb=0.4
def fv(x):
  x1=0.
  x2=lamb*(x[0]/20.)**2
  x3=lamb*(x[0]/20.)**2
  return np.array([x1,x2,x3])

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
