import sys
import os
import pdb
# sys.path.append(os.getcwd())
import numpy as np

#Grid = 'structuralGrid'
#K_a = 'Ka.npy'
#M_a = 'Ma.npy'
#M_a2 = 'Maa.npy'
#K_a2 = 'Kaa.npy'
model_name = '/Models/wingSP'
model = 'wingSP'
for i in range(len(os.getcwd().split('/'))):
    if os.getcwd().split('/')[-i-1] == 'FEM4INAS':
        feminas_dir = "/".join(os.getcwd().split('/')[0:-i])

#Grid = feminas_dir + model_name + '/FEM/' + Grid
#K_a = feminas_dir + model_name + '/FEM/' + K_a
#M_a = feminas_dir + model_name + '/FEM/' + M_a
#M_a2 = feminas_dir + model_name + '/FEM/' + M_a2
#K_a2 = feminas_dir + model_name + '/FEM/' + K_a2
#Ka=np.load(K_a)
#Ma=np.load(M_a)
#Ka2=np.load(K_aa)
#Ma2=np.load(M_aa)
#import scipy.linalg
#w,v=scipy.linalg.eigh(Ka,Ma)
#ww=np.sqrt(w)
NumModes = 53
NumBeams = 1
Nastran_modes = 0
ReplaceRBmodes = 0
Path4Phi2 = 1
#Nastran_modes=1
#op2name=os.getcwd()+'/103/n1b'

# Loading
if 'loads.py' in os.listdir(os.getcwd()):
  loading=1
else:
  loading=0

# Reading Grid file
#============================================
node_start = 1              # NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start # Number Nodes
start_reading = 3     # range(start_reading,len(lin)):
beam_start = 1           # j=int(s[4])-beam_start BeamSeg[j]
nodeorder_start = 1    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

RigidBody_Modes = 0
Clamped = 1
ClampX=[]
BeamsClamped=[]
if Clamped:
  BeamsClamped=[0]
  ClampX='np.array([6.214,0.,-0.2075])'


BeamConn = [[[],[]],[[],[]]]
#MBbeams = [0,1]
initialbeams = [0]
#MBdof = {0:[0,1,2], 1:[0,1,2]}
#MBnode = {0:None,1:0}
#MBnode2 = {0:2,1:3}
EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1,0,0])'
gravity_on=0
rotation_quaternions = 0
rotation_strains = 0
# ======================================================================================================
# Time Discretization
#======================================================================================================
t0=0;tf=15.;tn=15001
#init_q0=1
dynamic=1
static=0
linear=0
