import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/SailPlaneC/FEM/Maa.npy'
q0_file='q0_042.npy'
model_name='/Models/SailPlaneC'
Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/SailPlaneC/FEM/SP_GridC'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/SailPlaneC/FEM/Kaa.npy'
model='SailPlaneC'
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
###########################
# Read Grid File Settings #
###########################
start_reading=1
node_start=1
nodeorder_start=0
beam_start=0
#####################
# Topology Settings #
#####################
Nastran_modes=0
NumBeams=12
NumModes_res=0
NumModes=120
BeamConn=[[[2, 4], [6, 11], [3], [], [5], [], [7, 9], [8], [], [10], [], []], [[2, 4], [6, 11], [3], [], [5], [], [7, 9], [8], [], [10], [], []]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0.0
tn=10001
ti=np.linspace(0.0,5.0,10001)
tf=5.0
dt=0.0005
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
ClampX=np.array([14.,0.,1.4])
BeamsClamped=[0, 1]
RigidBody_Modes=0
Clamped=1
####################
# Loading Settings #
####################
loading=1
linear=0
dynamic=1
static=0
init_q0=1
