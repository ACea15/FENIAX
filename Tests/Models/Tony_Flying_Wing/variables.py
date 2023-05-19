import sys
import os
import pdb
# sys.path.append(os.getcwd())
import numpy as np

Grid='FWgrid.txt'
K_a='Kmat.npy'
M_a='Mmat.npy'

model_name='/'+'/'.join(os.getcwd().split('/')[-2:])
model=os.getcwd().split('/')[-1]
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])

Grid = feminas_dir + model_name+'/FEM/'+Grid
K_a = feminas_dir +model_name+'/FEM/'+K_a
M_a = feminas_dir +model_name+'/FEM/'+M_a


NumModes=30
NumBeams=7
Nastran_modes=0
Path4Phi2=1
ReplaceRBmodes=1
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
start_reading=1     #range(start_reading,len(lin)):
beam_start=0           # j=int(s[4])-beam_start BeamSeg[j]
nodeorder_start=0    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

RigidBody_Modes=1
Clamped=0
ClampX=[]
BeamsClamped=[]
if Clamped:
  BeamsClamped=[0]
  ClampX='np.array([0.,0.,0.])'


BeamConn = [[[1,4],[2,5],[3,6],[],[],[],[]],[[],[0,4],[1,5],[2,6],[0,1],[1,2],[2,3]]]
EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1,0,0])'


# ======================================================================================================
# Time Discretization
#======================================================================================================
t0=0;tf=1;tn=2

if (__name__ == '__main__'):


  saving=1
  if saving:
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_config
    #pdb.set_trace()
    write_config(locals())

  print('Running Variables')
