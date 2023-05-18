import numpy as np
import pdb

import  intrinsic.functions
import intrinsic.beam_path

def geometry_def(Grid,NumBeams,BeamConn,start_reading,beam_start,nodeorder_start,node_start,Clamped,ClampX,BeamsClamped):


    #Read Grid File
    #==============
    with open(Grid,'r') as f:
      lin=f.readlines()


    # Beam Segments structure:
    #========================
    class Structure():
      pass

    BeamSeg=[Structure() for i in range(NumBeams)]

    for i in range(NumBeams):
      BeamSeg[i].NodeX=[]           # Node Coordinates
      BeamSeg[i].NodeOrder=[]       # Node Order


    # Read Structural Grid file:
    #==========================
    for i in range(start_reading,len(lin)):
      s=lin[i].split()
      j=int(s[4])-beam_start
      BeamSeg[j].NodeX.append([float(s[0]),float(s[1]),float(s[2])])
      BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

    for i in range(NumBeams):
      BeamSeg[i].EnumNode=len(BeamSeg[i].NodeX)             # Number of nodes in the beam segment
    # Duplication of the nodes at the beam connections
    #=================================================

    NumNode= sum([len(BeamSeg[k].NodeX) for k in range(NumBeams)])

    if NumNode != max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start:
        raise ValueError(' Number of nodes not coincident between the NodeOrder and NodeX')

    NumNodes=NumNode  # Number of nodes after duplication

    if Clamped:

     for i in BeamsClamped:
       #DupNodes.append([BeamSeg[i].NodeOrder[-1],NumNodes])
       BeamSeg[i].NodeOrder.insert(0,NumNodes)
       BeamSeg[i].NodeX.insert(0,ClampX)
       NumNodes=NumNodes+1

    duplicate=1
    DupNodes=[]
    if duplicate:
      for i in range(NumBeams):
        for j in range(len(BeamConn[0][i])):
          DupNodes.append([BeamSeg[i].NodeOrder[-1],NumNodes])   # Add new node and its order
          BeamSeg[BeamConn[0][i][j]].NodeOrder.insert(0,NumNodes)   # Add the new node at the beginning of the  connected beam
          NumNodes=NumNodes+1
          BeamSeg[BeamConn[0][i][j]].NodeX.insert(0,BeamSeg[i].NodeX[-1])     # Add the coordinate
      DupNodes=np.array(DupNodes)


    Nodes=np.array([BeamSeg[i].NodeOrder for i in range(NumBeams)])


    # Geometry and reference system from the coordinates:
    #===================================================
    inverseconn={}
    for i in range(NumBeams):
      BeamSeg[i].NodeX=np.array(BeamSeg[i].NodeX)


    for i in range(NumBeams):
      BeamSeg[i].Conn=[[],[]]
      for j in range(2):
        BeamSeg[i].Conn[j]=BeamConn[j][i]
        if j==0:
            if BeamConn[j][i] == []:
                pass
            else:
               for lx in range(len(BeamConn[j][i])):
                 inverseconn[BeamConn[j][i][lx]]=i

      #pdb.set_trace()
      BeamSeg[i].EnumNodes=len(BeamSeg[i].NodeX)             # Number of nodes in the beam segment after duplication and clamped conditions

      if len(BeamSeg[i].NodeX)<2:
       BeamSeg[i].NodeDL=[np.linalg.norm(BeamSeg[i].NodeX-BeamSeg[BeamSeg[i].Conn[0][j]].NodeX[1]) for j in range(len(BeamSeg[i].Conn[0]))]
       continue

      BeamSeg[i].NodeDL=[np.linalg.norm(BeamSeg[i].NodeX[j+1]-BeamSeg[i].NodeX[j]) for j in range(len(BeamSeg[i].NodeX)-1)]        # Distance between subsequent nodes
      BeamSeg[i].NodeL=[sum(BeamSeg[i].NodeDL[0:j]) for j in range(len(BeamSeg[i].NodeDL)+1)]
      BeamSeg[i].L = np.linalg.norm(BeamSeg[i].NodeX[-1]-BeamSeg[i].NodeX[0])

    #pdb.set_trace()
    # Local Coordinate system
    #==========================

    BeamG=intrinsic.beam_path.Independent_Conn(NumBeams,BeamSeg)
    BeamG2=intrinsic.beam_path.Independent_Conn2(NumBeams,BeamSeg)
    #pdb.set_trace()

    for i in range(NumBeams):

     if len(BeamSeg[i].NodeX)<2:
      ic=i
      continue

     elif  len(BeamSeg[i].NodeX)<3:
       e1= BeamSeg[i].NodeX[1]-BeamSeg[i].NodeX[0]
     else:
       e1= BeamSeg[i].NodeX[2]-BeamSeg[i].NodeX[1]

     if BeamG2[i]==[]:

       if BeamG[i]==[]:
        e2=np.array([0,1,0])
        if abs(e1.dot(e2))/(np.linalg.norm(e1))>0.99999:
         e2=np.array([0,0,1])
       elif len(BeamSeg[BeamG[i]].NodeX)<3:
         e2= BeamSeg[BeamG[i]].NodeX[1]-BeamSeg[BeamG[i]].NodeX[0]
       else:
         e2= BeamSeg[BeamG[i]].NodeX[2]-BeamSeg[BeamG[i]].NodeX[1]
       BeamSeg[i].GlobalAxes=intrinsic.functions.Base2(e1,e2)
       continue
     elif len(BeamSeg[BeamG2[i]].NodeX)<3:
       e2= BeamSeg[BeamG2[i]].NodeX[1]-BeamSeg[BeamG2[i]].NodeX[0]
     else:
       e2= BeamSeg[BeamG2[i]].NodeX[2]-BeamSeg[BeamG2[i]].NodeX[1]

     BeamSeg[i].GlobalAxes=intrinsic.functions.Base(e1,e2)

    return BeamSeg, NumNode, NumNodes, DupNodes, inverseconn


def dispx(NumBeams):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for i in range(NumBeams):

      x = BeamSeg[i].NodeX[:,0]
      y = BeamSeg[i].NodeX[:,1]
      z = BeamSeg[i].NodeX[:,2]

      #ax.scatter(x, y, z, c='r', marker='o')
      ax.plot(x, y, z, c='r', marker='o')


    #plt.axis('off')

    #fig.suptitle('Mode'+str(modeplot)+'_Phi1:'+str(r))
    #plt.axis([0,80,-1,1,-3,3])
  plt.show()

print('Geometry Loaded')

if (__name__ == '__main__'):

   import importlib
   import Runs.Torun
   Runs.Torun.torun =  'Shell_Rafan'
   V=importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+'V')

   BeamSeg, NumNode, NumNodes, DupNodes,inverseconn= geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)

   print('Geometry Running')
