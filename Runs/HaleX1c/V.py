import numpy as np 

######################
# FEM Files Settings #
######################
feminas_dir='/Users/alvarocea/Imperial/Computations/FEM4INAS' #FEM4INAS directory
op2name='' #
M_a='/Users/alvarocea/Imperial/Computations/FEM4INAS/Models/HaleX1c/FEM/Maa.npy' #Mass matrix location
Grid='/Users/alvarocea/Imperial/Computations/FEM4INAS/Models/HaleX1c/FEM/structuralGrid.txt' #Grid file where ordered nodes and the beam-segments they belong to are located 
K_a='/Users/alvarocea/Imperial/Computations/FEM4INAS/Models/HaleX1c/FEM/Kaa.npy' #Stiffness matrix location
model='HaleX1c' #Model name
model_name='/Models/HaleX1c' #Directory to the model in Models
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
Nastran_modes=0 #Modes directly from Nastran
Path4Phi2=0 #0 -> Phi2 calculated in optimal way 1-> Path enforced in the opposite or 1-direction
ReplaceRBmodes=0 #Replace the rigid-body modes
NumBeams=1 #Number of beam-segments
NumModes_res=0 #Number of residualized modes
NumModes=5 #Number of modes in the analysis
BeamConn=[[[]], [[]]] #Connectivities between beam segments [[..[..0-direction..]..BeamNumber..],[..[..1-direction..]..BeamNumber..]]
Check_Phi2=0 #Check equilibrium of forces for Phi2 (in free-model) 
#################
# Time Settings #
#################
t0=0 #Initial time
tn=4000 #Total time steps
ti=np.linspace(0,50,4000) #Time vector
tf=50 #Final time
dt=0.0125031257814 #Increment of time
######################
# Constants Settings #
######################
I3=np.eye(3) #
g0=9.80665 #Gravity acceleration on earth
EMAT=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]]) #
e_1=np.array([1,0,0]) #Beam-segments local direction
#####################
# Boundary Settings #
#####################
MBnode={} #
MBdof={} #
ClampX=np.array([0.,0.,0.]) #Coordinate of clamped node
BeamsClamped=[0] #
RigidBody_Modes=0 #
MBdofree={} #
MBnode2={} #
initialbeams=[0] #Beam-segments attach to first node
MBbeams=[] #
Clamped=1 #Clamped model
#######################################
# Loading Settings to redirect solvers#
#######################################
loading=1 #0 or 1 whether a strcuctural force is defined in the analysis ()
linear=0 #0 or 1 for 'linear' analysis (removing Gammas, cubic terms, which is not exactly linear)
print_timeSteps=1 #Print time steps in the ODE solution
q0_file=None #File to function for Initial qs
dynamic=1 #0 or 1 for dynamic computatitions (nonlinear system of ODEs)
static=0 #0 or 1 for static computatitions (nonlinear system of algebraic equations)
init_q0=1 #Initial (qs) conditions other than 0
