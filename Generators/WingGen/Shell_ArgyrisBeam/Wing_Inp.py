import numpy as np
import os
#import pyNastran

model='Shell_ArgyrisBeam'

sol='103'
numloads=1
pch=1
# Geometry
#=============================================================
Lx=100.; Lyr=np.sqrt(20) ; Lyt=np.sqrt(20); Lzr=np.sqrt(20) ; Lzt=np.sqrt(20)
nx=26; ny=4; nz=4

dlx=Lx/(nx-1); dly=Lyr/(ny-1); dlz=Lzr/(nz-1)
tipy=0; tipz=0

# Shell  properties
# =============================================================
Em=2.1E+7
Nu=0.3
thickness=1./(4*np.sqrt(20))

# Condensation points
na=25
aset=np.linspace(dlx,Lx,na)

#Mass
Mass=224*np.ones(na)
I=np.zeros((6,na))
for i in range(na):
  I[0,i]=37.36;I[1,i]=0.;I[2,i]=18.68;I[3,i]=0.0;I[4,i]=0.0;I[5,i]=18.68
 #I11        I21       I22        I31        I32        I33

for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])   
femName = 'S_na%s.bdf' %(na)
