import numpy as np
import pyNastran
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
xx
op2name='S_0'
bdfname=op2name


model= OP2()
model.read_op2(op2name+'.op2')

#print(model.get_op2_stats())
#print(model.get_op2_stats(short=True))
#model.eigenvectors[1].modes
#model.eigenvectors[1]._times

eigenvectors=0
if eigenvectors:
    eig1=model.eigenvectors[1]
    eigen=eig1.data
    NumNodes = np.shape(eig1.data[0])[0]

displacement=1
if displacement:
    NumLoads=4
    u=[]
    for j in range(1,NumLoads+1):

      disp=model.displacements[j]    
      #u[op2name[k]].append(disp.data[itime,:,0:6])
      u.append(disp.data[-1])
      NumNodes = np.shape(u[0])[0]


fem = BDF(debug=True,log=None)
fem.cross_reference()
fem.read_bdf(bdfname+'.bdf',xref=True)


X=[]
for i in range(1,NumNodes+1):

    X.append(fem.Node(i).get_position())
X=np.asarray(X)


def displt(rt,k,dis):
  #scale=max([max(abs(rt[j][3*k:3*k+3])) for j in range(NumNodes)])
  scale=1.
  x=X[:,0]
  y=X[:,1]
  z=X[:,2]

  fig=plt.figure()
  figx = fig.add_subplot(111, projection='3d')
  figx.plot(x, y, z, c='r', marker='o',lw=0.7)
  u=-rt[:,k*3+0]/scale+x
  v=-rt[:,k*3+1]/scale+y
  w=rt[:,k*3+2]/scale+z

  if dis:
   figx.plot(x+u,y+v,z+w,c='b',lw=2)
  else:
   figx.plot(u,v,w,c='b',lw=2)
  #figx.axis('equal')
  #figx.set_xlim(0, 3); figx.set_ylim(-0.1, 0.1); figx.set_zlim(-0.1, 0.1)
  #figx.set_xlim(0, 10); figx.set_ylim(-1, 1); figx.set_zlim(-1, 1)
  plt.show()

#scale=max([max([max(abs(Phi0[ix][k][jx,:])) for jx in range(BeamSeg[ix].EnumNodes)]) for ix in range(NumBeams)])
#max([max(abs(eigen[0][j])) for j in range(NumNodes)])

