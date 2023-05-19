import numpy as np
import os
#import pyNastran
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])

model='wingCea'

sol='103'
static = 1
dynamic = 0
rotation = 0
free = 0
lumped = 0
pch=0
# Geometry
#=============================================================
#Lx=40.; Lyr=8. ; Lyt=2.; Lzr=0.16*Lyr ; Lzt=0.12*Lyt
# Lx=60.; Lyr=10. ; Lyt=1.5; Lzr=2. ; Lzt=0.5
# ###################################################################
# nx=21; ny=5; nz=5

# dlx=Lx/(nx-1); dly=Lyr/(ny-1); dlz=Lzr/(nz-1)
# tipy=-30.; tipz=0

############################
Lx=60.; Lyr=10.5 ; Lyt=2.5; Lzr=2. ; Lzt=0.8
nx=31; ny=7; nz=5

dlx=Lx/(nx-1); dly=Lyr/(ny-1); dlz=Lzr/(nz-1)
tipy=-25.; tipz=0


# Shell  properties
# =============================================================
Em=1E+10
Nu=0.3
thickness=0.3

# Condensation points
na=30
aset=np.linspace(dlx,Lx,na)

#Mass
rho = 2.5e3
# ym=(Lyr+Lyt)/2
# zm=(Lzr+Lzt)/2
# Am = (2*ym*thickness+2*zm*thickness)
# Im2 = 2*ym**3*thickness/12 + 2*ym**2/2*zm*thickness
# Im3 = 2*zm**3*thickness/12 + 2*zm**2/2*ym*thickness
# Im1 = Im2 + Im3
# Mass= rhom*Lx/na*Am*np.ones(na)
# I=np.zeros((6,na))
# for i in range(na):
#   I[0,i]=rhom*Lx/na*Im1;I[1,i]=0.;I[2,i]=rhom*Lx/na*Im2;I[3,i]=0.0;I[4,i]=0.0;I[5,i]=rhom*Lx/na*Im3
#  #I11        I21       I22        I31        I32        I33
femName = 'S_na%s.bdf' %(na)

#Loads
FORCE1=1
numLoads=6
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
  return 1.5*np.array(0.,1.,1.)

def fu(x):
  return 2*x

def fu(x):
  return 2*x

ti=2000.

for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='PyFem2NL2':
    pyfem2nl_dir="/".join(os.getcwd().split('/')[0:-i])
femName = 'Sa_%s.bdf' %(na)
