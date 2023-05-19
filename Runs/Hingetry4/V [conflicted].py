import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/media/pcloud/Computations/FEM4INAS/Models/Hingetry4/FEM/Ma.npy'
q0_file=''
model_name='/Models/Hingetry4'
Grid='/media/pcloud/Computations/FEM4INAS/Models/Hingetry4/FEM/structuralGrid'
K_a='/media/pcloud/Computations/FEM4INAS/Models/Hingetry4/FEM/Ka.npy'
model='Hingetry4'
feminas_dir='/media/pcloud/Computations/FEM4INAS'
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
Path4Phi2=1
ReplaceRBmodes=0
NumBeams=2
NumModes_res=0
NumModes=55
BeamConn=[[[1], []], [[], [0]]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0
tn=1601
ti=np.linspace(0,10.0,1601)
tf=10.0
dt=0.00625
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
MBnode={0: None, 1: 14}
MBbeams=[0, 1]
ClampX=[]
BeamsClamped=[]
RigidBody_Modes=1
MBdofree={0: [3, 4, 5], 1: [3, 4, 5]}
MBdof={0: [0, 1, 2], 1: [0, 1, 2]}
MBnode2={0: 20, 1: 21}
initialbeams=[0]
Clamped=0
####################
# Loading Settings #
####################
loading=1
linear=0
dynamic=1
static=0
init_q0=None
