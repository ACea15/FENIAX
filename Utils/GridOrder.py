import pdb
import numpy as np
import sys
import os
from pyNastran.bdf.bdf import BDF

 #WoF
 #=====
#Connections=[[[8,7,6,5,4,3,10,11,12,2]],[[1,83]],[29,33],[[34,27]],[28],[35,47],[69,75],[84],[50,54],[[55,48]],[49],[56,68],[76,82],[[13,14,25,26,15,16,17,18,19,20,21,22]],[23,24],[95,105],[106,116],[85,94]]
#structuralgrid='/structuralGrid_WF'


def Range(a,b):

  if b>a:
   y=range(a,b+1)

  elif a>b:
   y = [a-i for i in range(a+1-b)]
  else:
   y=[a]
  return y

def bdf_order_nodes(Connections,bdf_file,bdf=None,nodestart=1):
    """ Given the nodes defined in Connections and a structural grid file, returns
    the structure (BeamSeg) with location and order of the nodes.

    TODO: Being able to take non-consecutive nodes

    """
    #Connections=[[20,1],[21,23],[58,56],[36,55],[[24,59]],[25,34],[69,78],[60,68],[35]]
    #Connections=[[20,1],[21,23],[58,56],[36,55],[[24,69,35]],[25,34],[69,78],[59,68]]
    #Connections=[[22,20],[57,55],[[79,34]],[0,19],[35,54],[24,33],[68,77],[58,67]]

    NumBeams=len(Connections)
    if bdf == None:
      bdf=BDF(debug=True,log=None)
      bdf.read_bdf(bdf_file)
    # Beam Segments structure:
    #========================
    class Structure():
        pass


    BeamSeg=[Structure() for i in range(NumBeams)]

    beam2node={}
    disperse=[]
    #pdb.set_trace()
    for i in range(NumBeams):

        if np.size(Connections[i][0])>1:
         BeamSeg[i].NodeX=[[] for j in range(np.size(Connections[i][0]))]           # Node Coordinates
         BeamSeg[i].NodeOrder=[[] for j in range(np.size(Connections[i][0]))]
         disperse.append(i)
         beam2node[i]=[k for k in Connections[i][0]]
        else:
         if np.size(Connections[i])==1:
          BeamSeg[i].NodeX=[[]]           # Node Coordinates
          BeamSeg[i].NodeOrder=[[]]        # Node Order
          beam2node[i]=Connections[i]
         else:
          BeamSeg[i].NodeX=[[] for j in range(abs(Connections[i][0]-Connections[i][1])+1)]           # Node Coordinates
          BeamSeg[i].NodeOrder=[[] for j in range(abs(Connections[i][0]-Connections[i][1])+1)]      # Node Order
          beam2node[i]=Range(Connections[i][0],Connections[i][1])

    for segi in range(NumBeams):
      for j in range(len(beam2node[segi])):
        BeamSeg[segi].NodeX[j] = bdf.nodes[beam2node[segi][j]+nodestart].get_position()
        BeamSeg[segi].NodeOrder[j] = beam2node[segi][j]

    return BeamSeg


def grid_order_nodes(Connections,structuralgrid,NumNodes,nodestart,file_start = 3):
    """ Given the nodes defined in Connections and a structural grid file, returns
    the structure (BeamSeg) with location and order of the nodes.

    """
    #Connections=[[20,1],[21,23],[58,56],[36,55],[[24,59]],[25,34],[69,78],[60,68],[35]]
    #Connections=[[20,1],[21,23],[58,56],[36,55],[[24,69,35]],[25,34],[69,78],[59,68]]
    #Connections=[[22,20],[57,55],[[79,34]],[0,19],[35,54],[24,33],[68,77],[58,67]]

    NumBeams=len(Connections)
    # Beam Segments structure:
    #========================
    class Structure():
        pass


    BeamSeg=[Structure() for i in range(NumBeams)]

    beam2node={}
    disperse=[]

    for i in range(NumBeams):

        if np.size(Connections[i][0])>1:
         BeamSeg[i].NodeX=[[] for j in range(np.size(Connections[i][0]))]           # Node Coordinates
         BeamSeg[i].NodeOrder=[[] for j in range(np.size(Connections[i][0]))]
         disperse.append(i)
         beam2node[i]=[k for k in Connections[i][0]]
        else:
         if np.size(Connections[i])==1:
          BeamSeg[i].NodeX=[[]]           # Node Coordinates
          BeamSeg[i].NodeOrder=[[]]        # Node Order
          beam2node[i]=Connections[i]
         else:
          BeamSeg[i].NodeX=[[] for j in range(abs(Connections[i][0]-Connections[i][1])+1)]           # Node Coordinates
          BeamSeg[i].NodeOrder=[[] for j in range(abs(Connections[i][0]-Connections[i][1])+1)]      # Node Order
          beam2node[i]=Range(Connections[i][0],Connections[i][1])

    with open(structuralgrid,'r') as f:
        lin=f.readlines()

    for i in range(file_start,NumNodes+file_start):

      s=lin[i].split()
      Id=int(s[3])-nodestart
      for j in range(NumBeams):
        if Id in beam2node[j]:
         segi=j
         break
      if segi in disperse:
         for j in range(np.size(Connections[segi][0])):
           if Id==Connections[segi][0][j]:

            BeamSeg[segi].NodeX[j]=[float(s[0]),float(s[1]),float(s[2])]
            BeamSeg[segi].NodeOrder[j]=int(s[3])-nodestart

      else:
         BeamSeg[segi].NodeX[abs(Id-Connections[segi][0])]=[float(s[0]),float(s[1]),float(s[2])]
         BeamSeg[segi].NodeOrder[abs(Id-Connections[segi][0])]=int(s[3])-nodestart

    return BeamSeg

def write_structuralGrid_file(fname,BeamSeg,NumBeams):

  with open(fname,'w') as fd2:

   fd2.write("""TITLE = \"Reduced Structural Grid\ "VARIABLES = \"x\" \"y\" \"z\" \"id\" \"familynum\" \"forcenode\" \"displnode\" \n""")

  #VARIABLES = \"x\" \"y\" \"z\" \"id\" \"familynum\" \"forcenode\" \"displnode\" \"body\"
  #ZONE T=\"Structural nodes\" I=%i , J=1, K=1, F=POINT\n""" % (NumNodes))
   for i in range(NumBeams):
     for j in range(len(BeamSeg[i].NodeOrder)):
      fd2.write("%21.15e %21.15e %21.15e %i %i %i %i\n" % (BeamSeg[i].NodeX[j][0],BeamSeg[i].NodeX[j][1],BeamSeg[i].NodeX[j][2],BeamSeg[i].NodeOrder[j],i,1,1))
