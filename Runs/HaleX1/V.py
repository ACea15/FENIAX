import numpy as np 

######################
# FEM Files Settings #
######################
feminas_dir='/Users/alvarocea/Imperial/Computations/FEM4INAS'
op2name=''
M_a='/Users/alvarocea/Imperial/Computations/FEM4INAS/Models/HaleX1/FEM/Maa.npy'
q0_file=''
Grid='/Users/alvarocea/Imperial/Computations/FEM4INAS/Models/HaleX1/FEM/structuralGrid.txt'
K_a='/Users/alvarocea/Imperial/Computations/FEM4INAS/Models/HaleX1/FEM/Kaa.npy'
model='HaleX1'
model_name='/Models/HaleX1'
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
NumModes=5
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
loading=0
linear=0
dynamic=0
static=1
init_q0=None
