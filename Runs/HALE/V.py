import numpy as np 

######################
# FEM Files Settings #
######################
feminas_dir='/media/pcloud/Computations/FEM4INAS'
op2name=''
M_a='/media/pcloud/Computations/FEM4INAS/Models/HALE/FEM/Maa.npy'
q0_file=''
Grid='/media/pcloud/Computations/FEM4INAS/Models/HALE/FEM/structuralGrid.txt'
K_a='/media/pcloud/Computations/FEM4INAS/Models/HALE/FEM/Kaa.npy'
model='HALE'
model_name='/Models/HALE'
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
Path4Phi2=0
ReplaceRBmodes=0
NumBeams=1
NumModes_res=0
NumModes=10
BeamConn=[[[]], [[]]]
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
MBnode={0: None}
MBbeams=[0]
ClampX=[]
BeamsClamped=[]
RigidBody_Modes=0
MBdofree={0: [4]}
MBdof={0: [0, 1, 2, 3, 5]}
MBnode2={0: 15}
initialbeams=[0]
Clamped=0
####################
# Loading Settings #
####################
loading=0
linear=0
dynamic=0
static=1
init_q0=None
