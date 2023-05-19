import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/Users/alvarocea/Dropbox/Computations/FEM4INAS/Models/Shell_Rafan/FEM/Maa.npy'
q0_file='q0_002.npy'
model_name='/Models/Shell_Rafan'
Grid='/Users/alvarocea/Dropbox/Computations/FEM4INAS/Models/Shell_Rafan/FEM/structuralGrid'
K_a='/Users/alvarocea/Dropbox/Computations/FEM4INAS/Models/Shell_Rafan/FEM/Kaa.npy'
model='Shell_Rafan'
feminas_dir='/Users/alvarocea/Dropbox/Computations/FEM4INAS'
###########################
# Read Grid File Settings #
###########################
start_reading=3
node_start=1
nodeorder_start=500000
beam_start=1
#####################
# Topology Settings #
#####################
Nastran_modes=0
NumBeams=1
NumModes_res=0
NumModes=30
BeamConn=[[[]], [[]]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0
tn=5001
ti=np.linspace(0,20,5001)
tf=20
dt=0.004
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
linear=1
dynamic=1
static=0
init_q0=1
