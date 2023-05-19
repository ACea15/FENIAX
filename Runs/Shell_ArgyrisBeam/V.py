import numpy as np 

Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Shell_ArgyrisBeam/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Shell_ArgyrisBeam/FEM/Kaa.npy'
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Shell_ArgyrisBeam/FEM/Maa.npy'
op2name=''
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
model_name='/Models/Shell_ArgyrisBeam'
model='Shell_ArgyrisBeam'
node_start=1
start_reading=3
beam_start=1
nodeorder_start=500000
NumModes=100
NumBeams=1
BeamConn=[[[]], [[]]]
Nastran_modes=0
loading=1
t0=0
tf=1
tn=14
RigidBody_Modes=0
Clamped=1
ClampX=np.array([0.,0.,0.])
BeamsClamped=[0]
EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])
I3=np.eye(3)
e_1=np.array([1,0,0])
