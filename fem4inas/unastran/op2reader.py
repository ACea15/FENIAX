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

    def readModel(self, asets_only=False):

        if self.op2name:
            self.op2 = OP2()
            self.op2.set_additional_matrices_to_read({b'OPHP':False, b'OUG1':False})
            self.op2.read_op2(self.op2name)
            print(self.op2.get_op2_stats())
        if self.bdfname:
            self.fem = BDF(debug=True,log=None)
            self.fem.cross_reference()
            self.fem.read_bdf(self.bdfname,xref=True)
            print(self.fem.get_bdf_stats())
            if asets_only:
                self.nodes = sorted(self.fem.asets[0].node_ids)
                self.asets = 1
            else:
                self.nodes = sorted(self.fem.node_ids)
                self.asets=0

    def eigenvectors(self):

        eig1 = self.op2.eigenvectors[1]
        eigen = eig1.data
        NumNodes = np.shape(eig1.data[0])[0]
        #model.eigenvectors[1].modes
        #model.eigenvectors[1]._times

    def displacements(self):
        subcases = sorted(self.op2.displacements.keys())
        NumLoads = len(subcases)
        u = []
        t = []
        for j in subcases:

          disp=self.op2.displacements[j]
          #u[op2name[k]].append(disp.data[itime,:,0:6])
          if self.static:
              u.append(disp.data[-1])
          else:
              u.append(disp.data)
              t.append(disp.dts)
        u = np.array(u)
        t = np.array(t) 
        #NumAsets=op2.displacements[subcases[0]]._nnodes
        #asets = op2.displacements[1].node_gridtype[:,0]
        return t, u

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
