import numpy as np
# import sys
import os
# import pdb
# sys.path.append(os.getcwd())


Grid='structuralGrid'
K_a='Kaa.npy'
M_a='Maa.npy'

#model_name='/'+'/'.join(os.getcwd().split('/')[-2:])
model_name='/Models'+'/' + os.getcwd().split('/')[-1]
model=os.getcwd().split('/')[-1]
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])


Grid = feminas_dir + '/Models/'+model+'/FEM/'+Grid
K_a = feminas_dir+'/Models/' +model+'/FEM/'+K_a
M_a = feminas_dir+'/Models/' +model+'/FEM/'+M_a


NumModes=60
NumBeams=1
Nastran_modes=0
if Nastran_modes:
 op2name=['S_0']
else:
 op2name = ''

linear = 1
static = 1
dynamic=0
# Loading
if 'loads.py' in os.listdir(os.getcwd()):
  loading=1
else:
  loading=0


# Reading Grid file
#============================================
node_start=1              # NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start # Number Nodes
start_reading=3
beam_start=1           # j=int(s[4])-beam_start BeamSeg[j]
nodeorder_start=1    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

RigidBody_Modes=0
Clamped=1
if Clamped:
  BeamsClamped=[0]
  ClampX='np.array([0.,0.,0.])'


BeamConn=[[[]],[[]]]
EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1,0,0])'


# ======================================================================================================
# Time Discretization
#======================================================================================================
t0=0;tf=2*np.pi;tn=7
ti='10*np.pi*np.array([0.5,1.,4.,8.,12.,16.,20.])'

if (__name__ == '__main__'):

  saving=1
  if saving:
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_config
    write_config(locals())

  print('Running Variables')
