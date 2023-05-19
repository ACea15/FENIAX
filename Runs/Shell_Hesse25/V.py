import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Shell_Hesse25/FEM/Maa.npy'
q0_file=''
model_name='/Models/Shell_Hesse25'
Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Shell_Hesse25/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Shell_Hesse25/FEM/Kaa.npy'
model='Shell_Hesse25'
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
###########################
# Read Grid File Settings #
###########################
start_reading=3
node_start=1
nodeorder_start=499999
beam_start=1
#####################
# Topology Settings #
#####################
Nastran_modes=0
NumBeams=1
NumModes_res=0
NumModes=112
BeamConn=[[[]], [[]]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0.0
tn=1320
ti=np.linspace(0.0,10.0,1320)
tf=10.0
dt=0.00758150113723
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
ClampX=[]
BeamsClamped=[]
RigidBody_Modes=1
Clamped=0
####################
# Loading Settings #
####################
loading=1
linear=0
dynamic=1
static=0
init_q0=None
