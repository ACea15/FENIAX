import numpy as np
import os
import intrinsic.FEmodel
#import sys

Grid='structuralGrid'
K_a='Kaa.npy'
M_a='Maa.npy'

model_name='/'+'/'.join(os.getcwd().split('/')[-2:])
model=os.getcwd().split('/')[-1]
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])


Grid = feminas_dir + model_name+'/FEM/'+Grid
K_a = feminas_dir +model_name+'/FEM/'+K_a
M_a = feminas_dir +model_name+'/FEM/'+M_a


NumModes = 50#25*6#60# 6*25#35
NumBeams=18
#NumBeams=1
Nastran_modes=0
if Nastran_modes:
 op2name=['S_0']
else:
 op2name = ''

NumNode = 78
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
start_reading=1     #range(start_reading,len(lin)):
beam_start=0           # j=int(s[4])-beam_start BeamSeg[j]
nodeorder_start=0    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

RigidBody_Modes=0
Clamped=1
ClampX=[]
BeamsClamped=[]
if Clamped:
  BeamsClamped=[0,1]
  ClampX='np.array([14.,0.,1.4])'

#[[i] for i in range(1,50) ]
#BeamConn = [[[1],[2],[3],[4],[5],[6],[7],[8],[9],[]],[[],[0],[1],[2],[3],[4],[5],[6],[7],[8]]]

#BeamConn = [[[2,4],[6,11],[3],[],[5],[],[7,9],[8],[],[10],[],[]],[[2,4],[6,11],[3],[],[5],[],[7,9],[8],[],[10],[],[]]]
BeamConn = [[[2,7],[12,17],[3],[4],[5],[6],[],[8],[9],[10],[11],[],[13,15],[14],[],[16],[],[]],[[2,7],[6,12],[3],[4],[5],[6],[],[8],[9],[10],[11],[],[13,15],[14],[],[16],[],[]]]
#BeamConn = [[[]],[[]]]
EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1,0,0])'
initialbeams=[0,1]
dynamic=0
static=1
# ======================================================================================================
# Time Discretization
#======================================================================================================
t0=0.;tf=5.3;tn=6
#ti='np.asarray([2.,2.3,2.5,2.8,3.,3.5,4.,4.4,4.8,5.,5.3])'
ti='np.asarray([2.,2.5,3.,4.,4.8,5.3])'
#init_q0=1
#q0_file = 'q0_042.npy'
if (__name__ == '__main__'):

  saving=1
  if saving:
    import intrinsic.Tools.write_config_file
    from intrinsic.Tools.write_config_file import write_config
    write_config(locals())

  print('Running Variables')
