import numpy as np 

######################
# FEM Files Settings #
######################
feminas_dir='/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS' #FEM4INAS directory
op2name='/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/trim_gust1/../FEM/Dreal70.npy#/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/trim_gust1/../FEM/Vreal70.npy' #
K_a2='/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/FEM/Ka2.npy' #Stiffness matrix location for multibody system (zeros not removed)
M_a='/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/FEM/Ma.npy' #Mass matrix location
cg='/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/FEM/cg.npy' #Location of npy-file with calculated CG
M_a2='/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/FEM/Ma2.npy' #Mass matrix location for multibody system (zeros not removed)
Grid='/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/FEM/structuralGridc.txt' #Grid file where ordered nodes and the beam-segments they belong to are located 
K_a='/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/FEM/Ka.npy' #Stiffness matrix location
model='XRF1-trim2' #Model name
model_name='/Models/XRF1-trim2' #Directory to the model in Models
###############
# ODE solvers #
###############
ODESolver='RK4' #ODE solver for the dynamic system
###########################
# Read Grid File Settings #
###########################
start_reading=1 #range(start_reading,len(lin)):
node_start=1 #NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start
nodeorder_start=0 #aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start
beam_start=0 #j=int(s[4])-beam_start BeamSeg[j]
#####################
# Topology Settings #
#####################
Nastran_modes=1 #Modes directly from Nastran
Path4Phi2=1 #0 -> Phi2 calculated in optimal way 1-> Path enforced in the opposite or 1-direction
ReplaceRBmodes=0 #Replace the rigid-body modes
Nastran_modes_dic={} #Dictionary to put nastran modes into the current formulation
NumBeams=33 #Number of beam-segments
NumModes_res=0 #Number of residualized modes
NumModes=70 #Number of modes in the analysis
BeamConn=[[[1, 7, 13, 32], [2], [3], [4, 5], [28], [6], [], [8], [9], [10, 11], [30], [12], [], [14], [15], [16, 22], [17], [18, 24, 26], [19], [20], [21], [], [23], [], [25], [], [27], [], [29], [], [31], [], []], [[], [7, 13, 32, 0], [1], [2], [5, 3], [4, 3], [5], [1, 13, 32, 0], [7], [8], [11, 9], [10, 9], [11], [1, 7, 32, 0], [13], [14], [22, 15], [16], [24, 26, 17], [18], [19], [20], [16, 15], [22], [18, 26, 17], [24], [18, 24, 17], [26], [4], [28], [10], [30], [1, 7, 13, 0]]] #Connectivities between beam segments [[..[..0-direction..]..BeamNumber..],[..[..1-direction..]..BeamNumber..]]
Check_Phi2=0 #Check equilibrium of forces for Phi2 (in free-model) 
#################
# Time Settings #
#################
t0=0 #Initial time
tn=12001 #Total time steps
ti=np.linspace(0,57.0,12001) #Time vector
tf=57.0 #Final time
dt=0.00475 #Increment of time
######################
# Constants Settings #
######################
I3=np.eye(3) #
g0=[0.0, 0.0, -9.80665, 0.0, 0.0, 0.0] #Gravity acceleration on earth
EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]]) #
e_1=np.array([1,0,0]) #Beam-segments local direction
#####################
# Boundary Settings #
#####################
Constrains={} #Constrains={"c1":[[bodies],[beams],"1/0-constrain displacement,1/2/3 constrain x,y,z axes respectively"]}
NumBodies=1 #Total number of bodies
MBnode={0: None} #Position in the in the stiffness and mass matrices of node attach to MB node
aeromb=[] #Aero force input files of the multibody set
variablesmb=[] #Varibles input  files of the multibodyset
MBdof={0: [0, 1, 3, 5]} #Degrees of freedom joined in the Multibody (linear) system
Clamped=0 #Clamped model
forcesmb=[] #Force input files of the multibody set
ClampX=np.load('/mnt/ssd860/ACea/Imperial/Computations/FEM4INAS/Models/XRF1-trim2/FEM/cg.npy') #Coordinate of clamped node
BeamsClamped=[] #
RigidBody_Modes=2 #
MBdofree={0: [2, 4]} #Degrees of freedom independent in the Multibody (linear) system
NumConstrains=0 #Total number of constrains.
MBnode2={0: 0} #Position of multibody node in the stiffness and mass matrices
initialbeams=[0] #Beam-segments attach to first node
MBbeams=[0] #Multibody beams (in linear problem)
results_modesMB=[] #Folder to find in the multibody analysis 
#######################################
# Loading Settings to redirect solvers#
#######################################
gravity_on=1 #0 or 1 whether gravity is to be accounted for
loading=1 #0 or 1 whether a strcuctural force is defined in the analysis ()
linear=1 #0 or 1 for 'linear' analysis (removing Gammas, cubic terms, which is not exactly linear)
print_timeSteps=1 #Print time steps in the ODE solution
q0_file=None #File to function for Initial qs
dynamic=1 #0 or 1 for dynamic computatitions (nonlinear system of ODEs)
quadratic_integrals=1 #Quadratic terms in the integrals of the nonlinear terms Gamma1 and Gamma2
static=0 #0 or 1 for static computatitions (nonlinear system of algebraic equations)
init_q0=0 #Initial (qs) conditions other than 0
#######################
# Options for solvers #
#######################
rotation_strains=0 #
rotation_quaternions=1 #
