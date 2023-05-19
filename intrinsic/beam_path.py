import sys
import os
sys.path.append(os.getcwd())
import pdb
import numpy as np
from functions import *



def flatten(lis):
    l=list(lis)
    while type(max(l)) is list and len(max(l))>0:

       for i in range(len(l)):
         if type(l[i]) is list:
          if len(l[i])==0:
           continue
          for j in range(len(l[i])):
            l.insert(i+j,l[i+j][j])
          del l[i+j+1]
    return l

def flatten2(lis):
    l=list(lis)
    i=0
#while type(max(l)) is list:

    while i < len(l):
         if type(l[i]) is list:
          if len(l[i])==0:
           del l[i]
           continue
          for j in range(len(l[i])):
            l.insert(i+j,l[i+j][j])

          del l[i+j+1]
         else:
          i=i+1

    return l

def BeamTree(parentseg,grandpa2seg,BeamSeg):

    familyseg=[]
    while  np.size(parentseg)!=0:


      childseg=[]
      grandpa1seg=grandpa2seg
      grandpa2seg=[]

      if grandpa1seg==[]:
       for ngrandpa in range(np.size(grandpa1seg)+1):
        for ns in range(np.size(parentseg[ngrandpa])):

          if grandpa1seg==BeamSeg[parentseg[ngrandpa][ns]].Conn[0]:
            j=1
          elif  grandpa1seg==BeamSeg[parentseg[ngrandpa][ns]].Conn[1]:
            j=0


          childseg.append(BeamSeg[parentseg[ngrandpa][ns]].Conn[j])
          grandpa2seg=grandpa2seg+[parentseg[ngrandpa][ns]]
      else:
       for ngrandpa in range(np.size(grandpa1seg)):
        for ns in range(np.size(parentseg[ngrandpa])):

            if grandpa1seg[ngrandpa] in BeamSeg[parentseg[ngrandpa][ns]].Conn[0]:
               j=1
            elif  grandpa1seg[ngrandpa] in BeamSeg[parentseg[ngrandpa][ns]].Conn[1]:
               j=0

              #pdb.set_trace()
            childseg.append(BeamSeg[parentseg[ngrandpa][ns]].Conn[j])
            grandpa2seg=grandpa2seg+[parentseg[ngrandpa][ns]]

      parentseg=childseg
      familyseg=familyseg+childseg
    return flatten2(familyseg)

def BeamTreeOld(segi,BeamSeg):
  fg=1
  curseglist=[segi]
  totalchildseglist=[]
  while fg>0:
      nextseglist=[]
      for nss in range(np.size(curseglist)):
          totalchildseglist=totalchildseglist+BeamSeg[curseglist[nss]].Conn[0]
          nextseglist=nextseglist+BeamSeg[curseglist[nss]].Conn[0]

      curseglist=nextseglist
      if np.size(curseglist)==0:
          fg=0
  return totalchildseglist

def BeamTreeOld2(segi,BeamSeg):
  fg=1
  pdb.set_trace()
  curseglist=[segi]
  totalchildseglist=[]
  while fg>0:
      nextseglist=[]
      for nss in range(np.size(curseglist)):
          totalchildseglist=totalchildseglist+BeamSeg[curseglist[nss]].Conn[1]
          nextseglist=nextseglist+BeamSeg[curseglist[nss]].Conn[1]

      curseglist=nextseglist
      if np.size(curseglist)==0:
          fg=0
  return totalchildseglist

def Treesec(segi,BeamSeg):

    Tree=[[],[]]
    parentseg=[[segi]]
    for jx in range(2):
      if np.size(BeamSeg[segi].Conn[jx])>1:
        grandpa2seg=[BeamSeg[segi].Conn[jx][0]]
      else:
        grandpa2seg=BeamSeg[segi].Conn[jx]

      Tree[jx]=BeamTree(parentseg,grandpa2seg,BeamSeg)
    return Tree

def TreeOpt(Treex):
  if len(Treex[1])>len(Treex[0]):
     return Treex[0],0
  else:
     return Treex[1],1




def BeamTree2(parentseg,grandpa2seg,BeamSeg):

    familyseg=[]
    direc={}
    while  np.size(parentseg)!=0:


      childseg=[]
      grandpa1seg=grandpa2seg
      grandpa2seg=[]
      #pdb.set_trace()
      if grandpa1seg==[]:
       for ngrandpa in range(np.size(grandpa1seg)+1):
        for ns in range(np.size(parentseg[ngrandpa])):

          if grandpa1seg==BeamSeg[parentseg[ngrandpa][ns]].Conn[0]:
            j=1
          elif  grandpa1seg==BeamSeg[parentseg[ngrandpa][ns]].Conn[1]:
            j=0
          direc[parentseg[ngrandpa][ns]]=j
          #pdb.set_trace()
          childseg.append(BeamSeg[parentseg[ngrandpa][ns]].Conn[j])
          grandpa2seg=grandpa2seg+[parentseg[ngrandpa][ns]]
      else:
       for ngrandpa in range(np.size(grandpa1seg)):
        for ns in range(np.size(parentseg[ngrandpa])):

            if grandpa1seg[ngrandpa] in BeamSeg[parentseg[ngrandpa][ns]].Conn[0]:
               j=1
            elif  grandpa1seg[ngrandpa] in BeamSeg[parentseg[ngrandpa][ns]].Conn[1]:
               j=0
            direc[parentseg[ngrandpa][ns]]=j
            #pdb.set_trace()
            #pdb.set_trace()
            childseg.append(BeamSeg[parentseg[ngrandpa][ns]].Conn[j])
            grandpa2seg=grandpa2seg+[parentseg[ngrandpa][ns]]
       #pdb.set_trace()
      parentseg=childseg
      familyseg=familyseg+childseg

    return flatten2(familyseg),direc

def BeamTreeOposite(currentseg,BeamSeg):

    grandpaseg = BeamSeg[currentseg].Conn[0]
    if len(grandpaseg) > 1:
        grandpaseg = [grandpaseg[0]]
    familyseg,direc = BeamTree2([[currentseg]],grandpaseg,BeamSeg)

    return familyseg,direc
    
def Treesec2(segi,BeamSeg):
    """bbbbb"""
    Tree=[[],[]]
    direc=[[],[]]
    parentseg=[[segi]]
    for jx in range(2):
      if np.size(BeamSeg[segi].Conn[jx])>1:
        grandpa2seg=[BeamSeg[segi].Conn[jx][0]]
      else:
        grandpa2seg=BeamSeg[segi].Conn[jx]

      Tree[jx],direc[jx]=BeamTree2(parentseg,grandpa2seg,BeamSeg)
    return Tree,direc

def TreeOpt2(Treex,direc):
  if len(Treex[1])>len(Treex[0]):
     return Treex[0],direc[0]
  else:
     return Treex[1],direc[1]

def TreeOpt2_contrary(Treex,direc):
  if len(Treex[1])>len(Treex[0]):
     return Treex[1],direc[1]
  else:
     return Treex[0],direc[0]


def Independent_Conn(NumBeams,BeamSeg):

    BeamG=[[]for i in range(NumBeams)]
    for i in range(NumBeams):

        if len(BeamSeg[i].NodeX)<2:

          continue

        conn=BeamSeg[i].Conn
        bcon=flatten2(conn)

        for j in range(len(bcon)):

                  if  len(BeamSeg[i].NodeX)<3:
                   x= BeamSeg[i].NodeX[1]-BeamSeg[i].NodeX[0]
                  else:
                   x= BeamSeg[i].NodeX[2]-BeamSeg[i].NodeX[1]
                  if len(BeamSeg[bcon[j]].NodeX)<2:
                   continue
                  if len(BeamSeg[bcon[j]].NodeX)<3:
                   y= BeamSeg[bcon[j]].NodeX[1]-BeamSeg[bcon[j]].NodeX[0]
                  else:
                   y= BeamSeg[bcon[j]].NodeX[2]-BeamSeg[bcon[j]].NodeX[1]

                  if abs(x.dot(y))/(np.linalg.norm(x)*np.linalg.norm(y))>0.99999:
                     continue

                  else:
                   BeamG[i]=bcon[j]
                   break
    return BeamG


def Independent_Conn2(NumBeams,BeamSeg):

    BeamG=[[]for i in range(NumBeams)]
    for i in range(NumBeams):

        if len(BeamSeg[i].NodeX)<2:

          continue

        conn=BeamSeg[i].Conn[0]
        bcon=flatten2(conn)

        for j in range(len(bcon)):

                  if  len(BeamSeg[i].NodeX)<3:
                   x= BeamSeg[i].NodeX[1]-BeamSeg[i].NodeX[0]
                  else:
                   x= BeamSeg[i].NodeX[2]-BeamSeg[i].NodeX[1]
                  if len(BeamSeg[bcon[j]].NodeX)<2:
                   continue
                  if len(BeamSeg[bcon[j]].NodeX)<3:
                   y= BeamSeg[bcon[j]].NodeX[1]-BeamSeg[bcon[j]].NodeX[0]
                  else:
                   y= BeamSeg[bcon[j]].NodeX[2]-BeamSeg[bcon[j]].NodeX[1]

                  if abs(x.dot(y))/(np.linalg.norm(x)*np.linalg.norm(y))>0.99999:
                     continue

                  else:
                   BeamG[i]=bcon[j]
                   break
    return BeamG



print('Read beam_path')

if (__name__ == '__main__'):

    import importlib
    import  intrinsic.geometry
    import Runs.Torun
    Runs.Torun.torun = 'Tony_Flying_Wing'

    V=importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+'V')
    BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)
