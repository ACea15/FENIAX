import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Hingedfree/FEM/Maa.npy'
q0_file=''
model_name='/Models/Hingedfree'
Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Hingedfree/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Hingedfree/FEM/Kaa.npy'
model='Hingedfree'
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
###########################
# Read Grid File Settings #
###########################
start_reading=3
node_start=1
nodeorder_start=0
beam_start=0
#####################
# Topology Settings #
#####################
Nastran_modes=0
Path4Phi2=1
ReplaceRBmodes=0
NumBeams=2
NumModes_res=0
NumModes=15
BeamConn=[[[1], []], [[], [0]]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0
tn=2
ti=np.linspace(0,1,2)
tf=1
dt=1.0
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
MBnode={1: 0}
MBbeams=[1]
ClampX=np.array([0.,0.,0.])
BeamsClamped=[0]
RigidBody_Modes=1
MBdofree={1: [0, 1, 2, 3, 4, 5]}
MBdof={1: []}
MBnode2={1: 2}
initialbeams=[0]
Clamped=1
####################
# Loading Settings #
####################
loading=0
linear=0
dynamic=1
static=0
init_q0=None
