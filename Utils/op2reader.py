import numpy as np
import pyNastran
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class NastranReader:

    def __init__(self,op2name=None,bdfname=None,static=None):

        self.op2name =  op2name
        self.bdfname = bdfname
        self.static = static

    def readModel(self,Rop2=1,Rbdf=1):

        if self.op2name:
            self.op2 = OP2()
            self.op2.read_op2(self.op2name+'.op2')
            print(self.op2.get_op2_stats())
        if self.bdfname:
            self.fem = BDF(debug=True,log=None)
            self.fem.cross_reference()
            self.fem.read_bdf(self.bdfname+'.bdf',xref=True)
            print(self.fem.get_bdf_stats())
            try:
                self.nodes = sorted(self.fem.asets[0].node_ids)
                self.asets = 1
            except:
                self.nodes = sorted(self.fem.node_ids)
                self.asets=0


    def eigenvectors(self):

        eig1=op2.eigenvectors[1]
        eigen=eig1.data
        NumNodes = np.shape(eig1.data[0])[0]
        np.reshape(SRafa2.op2.eigenvectors[1].data[0],(1,6*25))
        #model.eigenvectors[1].modes
        #model.eigenvectors[1]._times

    def displacements(self):
        subcases = sorted(self.op2.displacements.keys())
        NumLoads = len(subcases)
        u=[]
        for j in subcases:

          disp=self.op2.displacements[j]
          #u[op2name[k]].append(disp.data[itime,:,0:6])
          if self.static:
              u.append(disp.data[-1])
          else:
              u.append(disp.data)

        u = np.array(u)
        #NumAsets=op2.displacements[subcases[0]]._nnodes
        #asets = op2.displacements[1].node_gridtype[:,0]
        return u

    def position(self):

        asets = self.op2.displacements[1].node_gridtype[:,0]
        if self.static:
            ti = sorted(self.op2.displacements.keys())
        else:
            ti = self.op2.displacements[1].dts
        # if self.bdfname:
        #     assert (asets == self.nodes).all(), 'Asets in FEM not equal to asets in OP2'

        X = self.geometry(aset=asets)
        u = self.displacements()

        r=[]
        for i in range(len(u)):
            if self.static:

                r.append(X + u[i][:,0:3])
            else:
                for tii in range(len(ti)):
                    r.append(X + u[i][tii,:,0:3])

        r = np.asanyarray(r)
        return ti,r


#mass, cg, I = fem.mass_properties()

    def geometry(self,condensed=None,full=None,aset=[]):

        if aset is not []:
            nodeIds = aset
        elif condensed:
            nodeIds = sorted(self.fem.asets[0].node_ids)
        elif full:
            nodeIds = sorted(self.fem.node_ids)

        X=[]
        for i in nodeIds:
            X.append(self.fem.Node(i).get_position())
        X=np.asarray(X)

        return X
    def plot_asets(self):
         from mpl_toolkits.mplot3d import Axes3D
         import matplotlib.pyplot as plt
         t,asetsX = self.position()
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         
         ax.scatter(asetsX[0,:,0], asetsX[0,:,1], asetsX[0,:,2], c='r', marker='.')

         ax.set_xlabel('X Label')
         ax.set_ylabel('Y Label')
         ax.set_zlabel('Z Label')

         plt.show()
# u=[]
# for i in subcases:
#     u.append(op2.displacements[i].data[-1][:NumAsets,0:3]+X)

# np.save('ut.npy',np.array(u))



# X=[]
# for i in range(1,NumNodes+1):

#     X.append(fem.Node(i).get_position())
# X=np.asarray(X)

# X=[]
# for i in range(1,26+1):

#     X.append(fem.Node(i).get_position())
# X=np.asarray(X)



def displt(rt,k,dis):
  #scale=max([max(abs(rt[j][3*k:3*k+3])) for j in range(NumNodes)])
  scale=1.
  x=X[:,0]
  y=X[:,1]
  z=X[:,2]

  fig=plt.figure()
  figx = fig.add_subplot(111, projection='3d')
  figx.plot(x, y, z, c='r', marker='o',lw=0.7)
  u=rt[:,k*3+0]/scale
  v=rt[:,k*3+1]/scale
  w=rt[:,k*3+2]/scale

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

# ut=model.displacements[1].data
# #ts=model.displacements[1].dts
# uy=ut[:,25,1]
# import matplotlib.pyplot as plt
# plt.plot(u[0,-1,:,0])
# #plt.axis([0, 6, 0, 20])
# plt.show()

# np.save('ut2.npy',np.array(u))


if (__name__ == '__main__'):

    pass
    # model = './Models/Shell_Rafan'
    # op2name = model+'/Wing400/nx26ny5nz5/S_na25'
    # bdfname = model+'/Wing400/nx26ny5nz5/S_na25'
    # SRafa = NastranReader(op2name,bdfname)

