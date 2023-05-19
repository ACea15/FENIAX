import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Simo45_5/FEM/Maa.npy'
q0_file=''
model_name='/Models/Simo45_5'
Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Simo45_5/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Simo45_5/FEM/Kaa.npy'
model='Simo45_5'
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
###########################
# Read Grid File Settings #
###########################
start_reading=3
node_start=1
nodeorder_start=1
beam_start=1
#####################
# Topology Settings #
#####################
Nastran_modes=0
NumBeams=5
NumModes_res=0
NumModes=30
BeamConn=[[[1], [2], [3], [4], []], [[], [0], [1], [2], [3]]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0
tn=31
ti=np.arange(0,3100,100)
tf=3000
dt=100.0
######################
# Constants Settings #
######################
I3=np.eye(3)
e_1=np.array([1.,0.,0.])
EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])
g=9.80665
#####################
# Boundary Settings #
#####################
ClampX=np.array([0.,0.,0.])
BeamsClamped=[0]
RigidBody_Modes=0
Clamped=1
####################
# Loading Settings #
####################
loading=1
linear=0
dynamic=0
static=1
init_q0=None
