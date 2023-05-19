import numpy as np 

######################
# FEM Files Settings #
######################
feminas_dir='/media/acea/work/projects/FEM4INAS' #FEM4INAS directory
op2name='' #
K_a2='' #Stiffness matrix location for multibody system (zeros not removed)
M_a='/media/acea/work/projects/FEM4INAS/Models/SailPlane/FEM/Maa.npy' #Mass matrix location
cg='' #Location of npy-file with calculated CG
M_a2='' #Mass matrix location for multibody system (zeros not removed)
Grid='/media/acea/work/projects/FEM4INAS/Models/SailPlane/FEM/structuralGrid' #Grid file where ordered nodes and the beam-segments they belong to are located 
K_a='/media/acea/work/projects/FEM4INAS/Models/SailPlane/FEM/Kaa.npy' #Stiffness matrix location
model='SailPlane' #Model name
model_name='/Models/SailPlane' #Directory to the model in Models
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
Nastran_modes_dic={} #Dictionary to put nastran modes into the current formulation
NumBeams=18 #Number of beam-segments
NumModes_res=0 #Number of residualized modes
NumModes=50 #Number of modes in the analysis
BeamConn=[[[2, 7], [12, 17], [3], [4], [5], [6], [], [8], [9], [10], [11], [], [13, 15], [14], [], [16], [], []], [[2, 7], [6, 12], [3], [4], [5], [6], [], [8], [9], [10], [11], [], [13, 15], [14], [], [16], [], []]] #Connectivities between beam segments [[..[..0-direction..]..BeamNumber..],[..[..1-direction..]..BeamNumber..]]
Check_Phi2=0 #Check equilibrium of forces for Phi2 (in free-model) 
#################
# Time Settings #
#################
t0=0.0 #Initial time
tn=6 #Total time steps
ti=np.asarray([2.,2.5,3.,4.,4.8,5.3]) #Time vector
tf=5.3 #Final time
dt=1.06 #Increment of time
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
MBnode={} #Position in the in the stiffness and mass matrices of node attach to MB node
aeromb=[] #Aero force input files of the multibody set
variablesmb=[] #Varibles input  files of the multibodyset
MBdof={} #Degrees of freedom joined in the Multibody (linear) system
Clamped=1 #Clamped model
forcesmb=[] #Force input files of the multibody set
ClampX=np.array([14.,0.,1.4]) #Coordinate of clamped node
BeamsClamped=[0, 1] #
RigidBody_Modes=0 #
MBdofree={} #Degrees of freedom independent in the Multibody (linear) system
NumConstrains=0 #Total number of constrains.
MBnode2={} #Position of multibody node in the stiffness and mass matrices
initialbeams=[0, 1] #Beam-segments attach to first node
MBbeams=[] #Multibody beams (in linear problem)
results_modesMB=[] #Folder to find in the multibody analysis 
#######################################
# Loading Settings to redirect solvers#
#######################################
gravity_on=0 #0 or 1 whether gravity is to be accounted for
loading=0 #0 or 1 whether a strcuctural force is defined in the analysis ()
linear=0 #0 or 1 for 'linear' analysis (removing Gammas, cubic terms, which is not exactly linear)
print_timeSteps=1 #Print time steps in the ODE solution
q0_file=None #File to function for Initial qs
dynamic=0 #0 or 1 for dynamic computatitions (nonlinear system of ODEs)
quadratic_integrals=0 #Quadratic terms in the integrals of the nonlinear terms Gamma1 and Gamma2
static=1 #0 or 1 for static computatitions (nonlinear system of algebraic equations)
init_q0=0 #Initial (qs) conditions other than 0
#######################
# Options for solvers #
#######################
rotation_strains=0 #
rotation_quaternions=0 #
