import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/media/pcloud/Computations/FEM4INAS/Models/Simo_Moment/FEM/Maa.npy'
q0_file=''
model_name='/Models/Simo_Moment'
Grid='/media/pcloud/Computations/FEM4INAS/Models/Simo_Moment/FEM/structuralGrid'
K_a='/media/pcloud/Computations/FEM4INAS/Models/Simo_Moment/FEM/Kaa.npy'
model='Simo_Moment'
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
Path4Phi2=0
ReplaceRBmodes=0
NumBeams=1
NumModes_res=0
NumModes=60
BeamConn=[[[]], [[]]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0
tn=7
ti=np.pi*np.array([0.5,1.,4.,8.,12.,16.,20.])
tf=6.28318530718
dt=1.0471975512
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
MBnode={}
MBbeams=[]
ClampX=np.array([0.,0.,0.])
BeamsClamped=[0]
RigidBody_Modes=0
MBdofree={}
MBdof={}
MBnode2={}
initialbeams=[0]
Clamped=1
####################
# Loading Settings #
####################
loading=1
linear=1
dynamic=0
static=1
init_q0=None
