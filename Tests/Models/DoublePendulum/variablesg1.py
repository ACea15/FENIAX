import sys
import os
import pdb
# sys.path.append(os.getcwd())
import numpy as np

Grid = 'structuralGrid'
K_a = 'Ka.npy'
M_a = 'Ma.npy'
M_a2 = 'Maa.npy'
K_a2 = 'Kaa.npy'
model_name = '/Models/DPendulum' #'/'+'/'.join(os.getcwd().split('/')[-2:])
model = 'DPendulum'#os.getcwd().split('/')[-1]
for i in range(len(os.getcwd().split('/'))):
    if os.getcwd().split('/')[-i-1] == 'FEM4INAS':
        feminas_dir = "/".join(os.getcwd().split('/')[0:-i])

Grid = feminas_dir + model_name + '/FEM/' + Grid
K_a = feminas_dir + model_name + '/FEM/' + K_a
M_a = feminas_dir + model_name + '/FEM/' + M_a
M_a2 = feminas_dir + model_name + '/FEM/' + M_a2
K_a2 = feminas_dir + model_name + '/FEM/' + K_a2
#Ka=np.load(K_a)
#Ma=np.load(M_a)
#Ka2=np.load(K_aa)
#Ma2=np.load(M_aa)
#import scipy.linalg
#w,v=scipy.linalg.eigh(Ka,Ma)
#ww=np.sqrt(w)
NumModes = 3
NumBeams = 1
ReplaceRBmodes = 0
Path4Phi2 = 1
Nastran_modes=1
Nastran_modes_dic = {1:range(6),2:range(6),3:[3,4,5],4:[3,4,5]}
op2folder=os.getcwd()+'/103/'
op2name='%sDreal3.npy#%sVreal3.npy' %(op2folder,op2folder)

# Loading
if 'loads.py' in os.listdir(os.getcwd()):
  loading=1
else:
  loading=0

# Reading Grid file
#============================================
node_start = 1              # NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start # Number Nodes
start_reading = 3     # range(start_reading,len(lin)):
beam_start = 1           # j=int(s[4])-beam_start BeamSeg[j]
nodeorder_start = 1    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

RigidBody_Modes = 1
Clamped = 0
ClampX=[]
BeamsClamped=[]
if Clamped:
  BeamsClamped=[0]
  ClampX='np.array([0.,0.,0.])'


BeamConn = [[[]],[[]]]
MBbeams = [0]
initialbeams = [0]
MBdof = {0:[0,1,2]}
MBnode = {0:None}
MBnode2 = {0:1}
EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1,0,0])'
gravity_on=1
rotation_quaternions = 1
rotation_strains = 0

# ======================================================================================================
# Multibody
#======================================================================================================
NumBodies=2
variablesmb=['V1','V2']
forcesmb=['F1','F2']
aeromb=[None,None]
results_modesMB = ['Results_modes1','Results_modes2']
NumConstrains = 1
Constrains={'c0':[[0,1],[0,0],'000',[[0,0,1.],[0,0,1.]],[1,2]]} # Constrains={'c1':[[bodies],[beams],'1/0-constrain displacement,1/2/3 constrain x,y,z axes respectively']}
# ======================================================================================================
# Time Discretization
#======================================================================================================
t0=0;tf=30.;tn=10001
init_q0=0
dynamic=1
static=0
linear=0
quadratic_integrals=1
if (__name__ == '__main__'):


  saving=1
  if saving:
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_config
    #pdb.set_trace()
    write_config(locals(),'V1')

  print('Running Variables')
