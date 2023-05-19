import numpy as np 

Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisFrame_20o/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisFrame_20o/FEM/Kaa.npy'
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisFrame_20o/FEM/Maa.npy'
op2name=''
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
model_name='/Models/ArgyrisFrame_20o'
model='ArgyrisFrame_20o'
node_start=1
start_reading=3
beam_start=1
nodeorder_start=1
NumModes=120
NumBeams=2
BeamConn=[[[1], []], [[], [0]]]
Nastran_modes=0
loading=1
t0=0
tf=1
tn=9
RigidBody_Modes=0
Clamped=1
ClampX=np.array([0.,0.,0.])
BeamsClamped=[0]
EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])
I3=np.eye(3)
e_1=np.array([1.,0.,0.])
