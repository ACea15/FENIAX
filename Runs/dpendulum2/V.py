import numpy as np 

######################
# FEM Files Settings #
######################
feminas_dir='/media/pcloud/Computations/FEM4INAS' #FEM4INAS directory
op2name='/media/pcloud/Computations/FEM4INAS/Models/dpendulum2/103/n1b8.npy' #
M_a='/media/pcloud/Computations/FEM4INAS/Models/dpendulum2/FEM/Ma.npy' #Mass matrix location
M_a2='/media/pcloud/Computations/FEM4INAS/Models/dpendulum2/FEM/Maa.npy' #Mass matrix location for multibody system (zeros not removed)
Grid='/media/pcloud/Computations/FEM4INAS/Models/dpendulum2/FEM/structuralGrid' #Grid file where ordered nodes and the beam-segments they belong to are located 
K_a='/media/pcloud/Computations/FEM4INAS/Models/dpendulum2/FEM/Ka.npy' #Stiffness matrix location
model='dpendulum2' #Model name
model_name='/Models/dpendulum2' #Directory to the model in Models
###############
# ODE solvers #
###############
ODESolver='RK4' #ODE solver for the dynamic system
###########################
# Read Grid File Settings #
###########################
start_reading=3 #range(start_reading,len(lin)):
node_start=1 #NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start
nodeorder_start=1 #aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start
beam_start=1 #j=int(s[4])-beam_start BeamSeg[j]
#####################
# Topology Settings #
#####################
Nastran_modes=1 #Modes directly from Nastran
Path4Phi2=1 #0 -> Phi2 calculated in optimal way 1-> Path enforced in the opposite or 1-direction
ReplaceRBmodes=0 #Replace the rigid-body modes
Nastran_modes_dic={1: [0, 1, 2, 3, 4, 5], 2: [0, 1, 2, 3, 4, 5], 3: [3, 4, 5], 4: [3, 4, 5]} #Dictionary to put nastran modes into the current formulation
NumBeams=2 #Number of beam-segments
NumModes_res=0 #Number of residualized modes
NumModes=8 #Number of modes in the analysis
BeamConn=[[[1], []], [[], [0]]] #Connectivities between beam segments [[..[..0-direction..]..BeamNumber..],[..[..1-direction..]..BeamNumber..]]
Check_Phi2=0 #Check equilibrium of forces for Phi2 (in free-model) 
#################
# Time Settings #
#################
t0=0 #Initial time
tn=25001 #Total time steps
ti=np.linspace(0,20.0,25001) #Time vector
tf=20.0 #Final time
dt=0.0008 #Increment of time
######################
# Constants Settings #
######################
I3=np.eye(3) #
g0=[0.0, -9.80665, 0.0, 0.0, 0.0, 0.0] #Gravity acceleration on earth
EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]]) #
e_1=np.array([1,0,0]) #Beam-segments local direction
#####################
# Boundary Settings #
#####################
MBnode={0: None, 1: 0} #
MBdof={0: [0, 1, 2], 1: [0, 1, 2]} #
ClampX=[] #Coordinate of clamped node
BeamsClamped=[] #
RigidBody_Modes=1 #
MBdofree={0: [3, 4, 5], 1: [3, 4, 5]} #
MBnode2={0: 2, 1: 3} #
initialbeams=[0] #Beam-segments attach to first node
MBbeams=[0, 1] #
Clamped=0 #Clamped model
#######################################
# Loading Settings to redirect solvers#
#######################################
gravity_on=1 #0 or 1 whether gravity is to be accounted for
loading=1 #0 or 1 whether a strcuctural force is defined in the analysis ()
linear=0 #0 or 1 for 'linear' analysis (removing Gammas, cubic terms, which is not exactly linear)
print_timeSteps=1 #Print time steps in the ODE solution
q0_file=None #File to function for Initial qs
dynamic=1 #0 or 1 for dynamic computatitions (nonlinear system of ODEs)
quadratic_integrals=1 #Quadratic terms in the integrals of the nonlinear terms Gamma1 and Gamma2
static=0 #0 or 1 for static computatitions (nonlinear system of algebraic equations)
init_q0=0 #Initial (qs) conditions other than 0
#######################
# Options for solvers #
#######################
rotation_strains=1 #
rotation_quaternions=0 #
