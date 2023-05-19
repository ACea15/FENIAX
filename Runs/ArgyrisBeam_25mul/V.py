import numpy as np 

Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisBeam_25mul/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisBeam_25mul/FEM/Kaa.npy'
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisBeam_25mul/FEM/Maa.npy'
op2name=''
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
model_name='/Models/ArgyrisBeam_25mul'
model='ArgyrisBeam_25mul'
node_start=1
start_reading=3
beam_start=1
nodeorder_start=1
NumModes=100
NumBeams=3
BeamConn=[[[1], [2], []], [[], [0], [1]]]
Nastran_modes=0
loading=1
t0=0
tf=1
tn=11
RigidBody_Modes=0
Clamped=1
ClampX=np.array([0.,0.,0.])
BeamsClamped=[0]
EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])
I3=np.eye(3)
e_1=np.array([1,0,0])
