import numpy as np
import os
#import sys

Grid='structuralGrid'
K_a='Kaa.npy'
M_a='Maa.npy'

model_name='/Models/ArgyrisFrame_20'#'/'+'/'.join(os.getcwd().split('/')[-2:])
model='ArgyrisFrame_20'#os.getcwd().split('/')[-1]
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])


Grid = feminas_dir + model_name+'/FEM/'+Grid
K_a = feminas_dir +model_name+'/FEM/'+K_a
M_a = feminas_dir +model_name+'/FEM/'+M_a


NumModes = 120#50*6
NumBeams=2
Nastran_modes=0
if Nastran_modes:
 op2name=['S_0']
else:
 op2name = ''


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
nodeorder_start=1    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

RigidBody_Modes=0
Clamped=1
ClampX=[]
BeamsClamped=[]
if Clamped:
  BeamsClamped=[0]
  ClampX='np.array([0.,0.,0.])'

#[[i] for i in range(1,50) ]
#BeamConn = [[[1],[2],[3],[4],[5],[6],[7],[8],[9],[]],[[],[0],[1],[2],[3],[4],[5],[6],[7],[8]]]

BeamConn = [[[1],[]],[[],[0]]]

EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1.,0.,0.])'

dynamic=0
static=1

# ======================================================================================================
# Time Discretization
#======================================================================================================
t0=0;tf=1040;tn=6
#ti='np.asarray([50.,75,100,150,200,250,300,400,500,550,600])'
#ti='np.asarray([50,75.,100,200.,300,350,430,600,700,850,1040])'
ti='np.asarray([50,100,300,430,700,1040])'
if (__name__ == '__main__'):

  saving=1
  if saving:
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_config
    write_config(locals())

  print('Running Variables')
