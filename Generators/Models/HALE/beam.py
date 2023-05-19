import numpy as np
import os
from Generators.beam_inp0 import *
# Main directory
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])

# Solution in Nastran
sol='103'
# Analysis type
static=1
dynamic=0
# Extract matrices in sol 103
pch=0
# Number of beams
NumBeams=1
# Clamp the model (needed in static solution)
Clamped=1
# Constraints
spc =[2]
# Nodes clamped: the first one of each beam clamped
BeamsClamped=[0]
# Coord. of the clamped node
Node0=np.array([0.,0.,0.])
# Connectivities of beams
BeamConn=[0]
# Model name
model='HaleX12'
# Length of each beam
L=[16. for i in range(NumBeams)]
# Number of nodes in each beam
N=[1+16]+[1 for i in range(NumBeams-1)]
# Width of beam
W=[1. for i in range(NumBeams)]
# Height of beam
H=[0.3831236556e-1 for i in range(NumBeams)]
# Thickness of beam
TH=[3e-3 for i in range(NumBeams)]
g1=1.
g2=5.
# Young modulus of beam
E=[g2*1e6/(TH[0]*(1./6+H[0]/2)) for i in range(NumBeams)]
# Area of beam
# Poisson's ratio of beam
NU=[0.3 for i in range(NumBeams)]
G = E[0]/(2*(1+NU[0]))
# Moment of inertia of beam, xx,zz,yy,zy
J=[g1*1e4/G for i in range(NumBeams)]
I1=[TH[0]*(H[0]**3/6+H[0]**2/2) for i in range(NumBeams)]
I2=[TH[0]*(1./6+H[0]/2) for i in range(NumBeams)]
I12=[0. for i in range(NumBeams)]
# transverse shear stiffness
K1=[None for i in range(NumBeams)]
K2=[None for i in range(NumBeams)]
# x-direction of each beam
Direc=[np.asarray([0,1,0]) for i in range(NumBeams)]
# y-or-z cross-sectional direction
beamcross = [np.asarray([0.,0.,1.]) for i in range(NumBeams)]
# property id
PID=[1 for i in range(NumBeams)]
# density of each beam
rho=[0.75 for i in range(NumBeams)]
I1m=[0.1 for i in range(NumBeams)]
I2m=[0.05 for i in range(NumBeams)]
I3m=[0.05 for i in range(NumBeams)]
#Area=[rho[0]*L[0]*(I1[0]+I2[0])/(Im*L[0]) for i in range(NumBeams)]
# Lumped masses, conm2 or conm1
conm2=1
conm1=0
if conm1:
  pass
if conm2:
  mass = [[rho[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I11 = [[I1m[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I22 = [[I2m[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I33 = [[I3m[i]*L[i]/(N[i]) for j in range(N[i])] for i in range(NumBeams)]
  I21 = I31 = I32 = [[0. for j in range(N[i])] for i in range(NumBeams)]
  X1 = X2 = X3 =  [[0. for j in range(N[i])] for i in range(NumBeams)]
  mass[0][-1] = mass[0][-1]/2
  I11[0][-1] = I11[0][-1]/2
  I22[0][-1] = I22[0][-1]/2
  I33[0][-1] = I33[0][-1]/2
