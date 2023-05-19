import numpy as np
import os
# import sys

Grid='structuralgrid'
K_a='Kaa.npy'
M_a='Maa.npy'

model_name='/Models'+'/' + os.getcwd().split('/')[-1]
model=os.getcwd().split('/')[-1]
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])


Grid = feminas_dir + '/Models/'+model+'/FEM/'+Grid
K_a = feminas_dir+'/Models/' +model+'/FEM/'+K_a
M_a = feminas_dir+'/Models/' +model+'/FEM/'+M_a




NumModes = 150#60#150#25*6#60# 6*25#35
NumBeams=1
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

BeamConn = [[[]],[[]]]

EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1,0,0])'


# ======================================================================================================
# Time Discretization
#======================================================================================================
#ti='np.asarray([1.,3.7,7.6,12.1,15.5,17.5,25.2,39.3,48.2,61.,80.,94.5,109.5,120.])'
ti='np.asarray([3.7,12.1,17.5,39.3,61.,94.5,120.])'
#ti='np.asarray([1.,3.7,7.6,12.1,15.5,17.5,25.2,39.3,48.2,61.,80.,94.5,109.5,120.])'
#ti='np.asarray([6.,15.,20.,25.,30.,35.,40.,50.,60.,65.,70.,75.,80.])'
t0 = 0;tf = 120.;tn = 7
dynamic=0
static=1


if (__name__ == '__main__'):

  saving=1
  if saving:
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_config
    write_config(locals())



  print('Running Variables')
