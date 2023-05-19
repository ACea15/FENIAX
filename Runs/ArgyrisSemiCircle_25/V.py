import numpy as np 

Grid='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisSemiCircle_25/FEM/structuralGrid'
K_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisSemiCircle_25/FEM/Kaa.npy'
M_a='/home/ac5015/Dropbox/Computations/FEM4INAS/Models/ArgyrisSemiCircle_25/FEM/Maa.npy'
op2name=''
feminas_dir='/home/ac5015/Dropbox/Computations/FEM4INAS'
model_name='/Models/ArgyrisSemiCircle_25'
model='ArgyrisSemiCircle_25'
node_start=1
start_reading=3
beam_start=1
nodeorder_start=1
NumModes=60
NumBeams=25
BeamConn=[[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], []], [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23]]]
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
e_1=np.array([1,0,0])
