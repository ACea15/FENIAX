import numpy as np 

######################
# FEM Files Settings #
######################
op2name=''
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ShellA380_15/FEM/Mi.npy'
q0_file='q0_2-2.npy'
model_name='/Models/ShellA380_15'
Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ShellA380_15/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ShellA380_15/FEM/Ki.npy'
model='ShellA380_15'
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
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
NumModes=90
BeamConn=[[[]], [[]]]
Check_Phi2=0
#################
# Time Settings #
#################
t0=0.0
tn=4001
ti=np.linspace(0.0,10.0,4001)
tf=10.0
dt=0.0025
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
dynamic=1
static=0
init_q0=0
