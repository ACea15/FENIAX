import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/HesseU_31/FEM/Maa.npy'
q0_file=''
model_name='/Models/HesseU_31'
Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/HesseU_31/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/HesseU_31/FEM/Kaa.npy'
model='HesseU_31'
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
###########################
# Read Grid File Settings #
###########################
start_reading=3
node_start=1
nodeorder_start=0
beam_start=1
#####################
# Topology Settings #
#####################
Nastran_modes=0
NumBeams=3
NumModes=45
BeamConn=[[[1], [2], []], [[], [0], [1]]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0
tn=6000
ti=np.linspace(0,20,6000)
tf=20
dt=0.0033338889815
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
