import numpy as np
import os
import sys
import  intrinsic.FEmodel
import pdb

Grid='structuralGrid'
K_a='Kaa.npy'
M_a='Maa.npy'

model_name='/Models/Hesse_25'
model='Hesse_25'
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])


Grid = feminas_dir + model_name+'/FEM/'+Grid
K_a = feminas_dir +model_name+'/FEM/'+K_a
M_a = feminas_dir +model_name+'/FEM/'+M_a



NumModes = 150
NumModes_res = 0
NumBeams=1
Nastran_modes=0
if Nastran_modes:
 op2name=['S_0']
else:
 op2name = ''

NumNode = 25
Ka,Ma,Dreal,Vreal=intrinsic.FEmodel.fem(K_a,M_a,0,0,NumNode,NumModes)
w=np.sqrt(Dreal)
# Loading
if 'loads.py' in os.listdir(os.getcwd()):
  loading=1
else:
  loading=0

# Reading Grid file
#============================================
node_start=1              # NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start # Number Nodes
start_reading=3     #range(start_reading,len(lin)):
beam_start=1           # j=int(s[4])-beam_start BeamSeg[j]
nodeorder_start=0    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

RigidBody_Modes=1
Clamped=0
ClampX=[]
BeamsClamped=[]
if Clamped:
  BeamsClamped=[0]
  ClampX='np.array([0.,0.,0.])'

#[[i] for i in range(1,50) ]
#BeamConn = [[[1],[2],[3],[4],[5],[6],[7],[8],[9],[]],[[],[0],[1],[2],[3],[4],[5],[6],[7],[8]]]

BeamConn = [[[]],[[]]]

EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1.,0.,0.])'

dynamic=1
static=0

# ======================================================================================================
# Time Discretization
#======================================================================================================
t0=0.;tf=10.;tn=6401#501#6401#2872#2001#6401

if (__name__ == '__main__'):

  saving=1
  if saving:
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_config
    #pdb.set_trace()
    write_config(locals())

  print('Running Variables')
