import numpy as np 

Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Gerandin_45/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Gerandin_45/FEM/Kaa.npy'
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/Gerandin_45/FEM/Maa.npy'
op2name=''
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
model_name='/Models/Gerandin_45'
model='Gerandin_45'
node_start=1
start_reading=3
beam_start=1
nodeorder_start=1
NumModes=60
NumBeams=10
BeamConn=[[[1], [2], [3], [4], [5], [6], [7], [8], [9], []], [[], [0], [1], [2], [3], [4], [5], [6], [7], [8]]]
Nastran_modes=0
loading=1
t0=0
tf=1
tn=20
RigidBody_Modes=0
Clamped=1
ClampX=np.array([0.,0.,0.])
BeamsClamped=[0]
EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])
I3=np.eye(3)
e_1=np.array([1,0,0])
