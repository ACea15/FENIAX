import numpy as np
#import pyNastran
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import pdb
import os

mesh=BDF(debug=True,log=None)

import importlib
from Run import torun
inp=importlib.import_module("Generators"+'.'+'BeamGen'+'.'+torun+'.'+'BGen_Inp')
reload(inp)
# Beam Segmexonts structure:
#========================

class Structure():
  pass

global BeamSeg
BeamSeg=[Structure() for i in range(inp.NumBeams)]


for i in range(inp.NumBeams):

 BeamSeg[i].Lx=inp.L[i]
 BeamSeg[i].nx=inp.N[i]
 try:
   BeamSeg[i].w=inp.W[i]
   BeamSeg[i].h=inp.H[i]
   BeamSeg[i].w=inp.W[i]
   BeamSeg[i].th=inp.TH[i]
 except AttributeError:
   print

 BeamSeg[i].nu=inp.NU[i]
 BeamSeg[i].e=inp.E[i]
 BeamSeg[i].j=inp.J[i]
 BeamSeg[i].area=inp.Area[i]
 BeamSeg[i].I1=inp.I1[i]
 BeamSeg[i].I2=inp.I2[i]
 BeamSeg[i].K1=inp.K1[i]
 BeamSeg[i].K2=inp.K2[i]
 BeamSeg[i].direc=inp.Direc[i]/np.linalg.norm(inp.Direc[i])
 if inp.Clamped and i in inp.BeamsClamped:
  BeamSeg[i].dl=BeamSeg[i].Lx/(BeamSeg[i].nx-1)
 elif inp.Clamped==0 and i==0:
  BeamSeg[i].dl=BeamSeg[i].Lx/(BeamSeg[i].nx-1)
 else:
  BeamSeg[i].dl=BeamSeg[i].Lx/(BeamSeg[i].nx)

 BeamSeg[i].idmat=i+1
 BeamSeg[i].idpbeam=i+1
 BeamSeg[i].pid=inp.PID[i]
 BeamSeg[i].NodeX=np.zeros((BeamSeg[i].nx,3))
 BeamSeg[i].rho=inp.rho[i]



if inp.Clamped:
  for i in range(inp.NumBeams):
      for j in range(BeamSeg[i].nx):

         if i in inp.BeamsClamped:
           if j==0:
             x0=inp.Node0
             BeamSeg[i].NodeX[j] = x0
           else:
             x0=BeamSeg[i].NodeX[j-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl
         else:
           if j==0:
             x0=BeamSeg[inp.BeamConn[i]].NodeX[-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl
           #elif j==BeamSeg[i].nx-1:
           #  continue
           else:
             x0=BeamSeg[i].NodeX[j-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl

else:
  count=0
  for i in range(inp.NumBeams):
      BeamSeg[i].NodeOrder = []
      for j in range(BeamSeg[i].nx):
         BeamSeg[i].NodeOrder.append(count)
         count +=count
         if i==0:
           if j==0:
             x0=inp.Node0
             BeamSeg[i].NodeX[j] = x0
           else:
             x0=BeamSeg[i].NodeX[j-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl
         else:
           if j==0:
             x0=BeamSeg[inp.BeamConn[i]].NodeX[-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl
           #elif j==BeamSeg[i].nx-1:
           #  continue
           else:
             x0=BeamSeg[i].NodeX[j-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl


NumNodes=sum(inp.N)

BeamSeg[i].M=0
BeamSeg[i].C=0

Kmat=np.zeros((NDof,NDof))
#pdb.set_trace()
for i in range(Numbeam):

    # First obtain transformation matrix    # GlobalAxes*local=global         # each BeamSegment is an element      # Kmat is ordered by node, *6

    BeamSegC=Rot6(BeamSeg[i].GlobalAxes)
    for j in range(BeamSeg[i].EnumNode-1):
        # for i=1:1        # start and end of beam element
        baseidstart=(BeamSeg[i].NodeOrder[j])*6
        baseidend=(BeamSeg[i].NodeOrder[j+1])*6
        fstartmat,fendmat,fstartmatr,fendmatr=beamsectioncoeff(np.max(BeamSeg[i].NodeL)/(len(BeamSeg[i].NodeL)-1),BeamSeg[i].C)
        # start and end converts a set displacement at the end of the  BeamSegment to the equivalent forces

        # first, displacement at the end node        # localforce=fendmat*localdisplacement        # globalforce=ownc*localforce        # localdisp=ownc'*globaldisp
        fendmatg=BeamSegC.dot(fendmat).dot((BeamSegC).T)
        fstartmatg=BeamSegC.dot(fstartmat).dot(BeamSegC.T)
        Kmat[baseidend:baseidend+6,baseidend:baseidend+6]=Kmat[baseidend:baseidend+6,baseidend:baseidend+6]+fendmatg
        Kmat[baseidstart:baseidstart+6,baseidend:baseidend+6]=Kmat[baseidstart:baseidstart+6,baseidend:baseidend+6]+fstartmatg

        # then, displacement at the start node follows the same process
        fendmatgr=BeamSegC.dot(fendmatr).dot(BeamSegC.T)
        fstartmatgr=BeamSegC.dot(fstartmatr).dot(BeamSegC.T)
        Kmat[baseidend:baseidend+6,baseidstart:baseidstart+6]=Kmat[baseidend:baseidend+6,baseidstart:baseidstart+6]+fendmatgr
        Kmat[baseidstart:baseidstart+6,baseidstart:baseidstart+6]=Kmat[baseidstart:baseidstart+6,baseidstart:baseidstart+6]+fstartmatgr
