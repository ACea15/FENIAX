import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/SailPlaneCnew/FEM/Maa.npy'
q0_file=''
model_name='/Models/SailPlaneCnew'
Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/SailPlaneCnew/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/SailPlaneCnew/FEM/Kaa.npy'
model='SailPlaneCnew'
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
NumBeams=18
NumModes_res=0
NumModes=50
BeamConn=[[[2, 7], [12, 17], [3], [4], [5], [6], [], [8], [9], [10], [11], [], [13, 15], [14], [], [16], [], []], [[2, 7], [6, 12], [3], [4], [5], [6], [], [8], [9], [10], [11], [], [13, 15], [14], [], [16], [], []]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0.0
tn=6
ti=np.asarray([2.,2.5,3.,4.,4.8,5.3])
tf=5.3
dt=1.06
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
dynamic=0
static=1
init_q0=None
