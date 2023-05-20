import numpy as np
import os
import  intrinsic.FEmodel
# import sys

Grid='structuralGrid'
K_a='Kaa.npy'
M_a='Maa.npy'

model_name='/'+'/'.join(os.getcwd().split('/')[-2:])
model=os.getcwd().split('/')[-1]
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='FEM4INAS':
    feminas_dir="/".join(os.getcwd().split('/')[0:-i])

Grid = feminas_dir + model_name+'/FEM/'+Grid
K_a = feminas_dir +model_name+'/FEM/'+K_a
M_a = feminas_dir +model_name+'/FEM/'+M_a


NumModes = 12#150#70#150#35#6*24#35
NumBeams=1
Nastran_modes=0
if Nastran_modes:
 op2name=['S_0']
else:
 op2name = ''

NumNode = 10
Ka,Ma,Dreal,Vreal=intrinsic.FEmodel.fem(K_a,M_a,0,0,NumNode,NumModes)
w=np.sqrt(Dreal)

# Loading
if 'loads.py' in os.listdir(os.getcwd()):
  loading=1
else:
  loading=0

# Reading Grid file
#============================================
node_start=1              # NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start # Number Nodes
start_reading=3     #range(start_reading,len(lin)):
beam_start=0           # j=int(s[4])-beam_start BeamSeg[j]
nodeorder_start=00000    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start)

RigidBody_Modes=0
Clamped=1
ClampX=[]
BeamsClamped=[]
if Clamped:
  BeamsClamped=[0]
  ClampX='np.array([2.,0.,0.166667])'

#[[i] for i in range(1,50) ]
#BeamConn = [[[1],[2],[3],[4],[5],[6],[7],[8],[9],[]],[[],[0],[1],[2],[3],[4],[5],[6],[7],[8]]]

BeamConn = [[[]],[[]]]

EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1,0,0])'
if Clamped:
  Check_Phi2 = 0
else:
  Check_Phi2 = 1

# ====================================================================================================
# Time Discretization
#=====================================================================================================
t0=0.;tf=30.;tn=30000
#dt = float(tf-t0)/(tn-1)

#init_q0=0
#q0_file = 'q0_04-04.npy'
dynamic=1
static=0
linear=1


if (__name__ == '__main__'):

  saving=1
  if saving:
    # femL=['Grid','K_a','M_a','op2name','feminas_dir','model_name','model','q0_file']
    # topL=['NumModes','NumBeams','BeamConn','Nastran_modes','loading','static','dynamic','init_q0','linear','t0','tf','tn','dt']
    # readL=['node_start','start_reading','beam_start','nodeorder_start']
    # boundaryL=['RigidBody_Modes','Clamped','ClampX','BeamsClamped','Check_Phi2']
    # constantL=['EMAT','I3','e_1']

    # if not os.path.exists(feminas_dir + '/Runs/'+model):
    #   os.makedirs(feminas_dir + '/Runs/'+model)
    #   with open(feminas_dir + '/Runs/'+model+'/__init__.py', 'w') as f2:
    #     print('Init file created')

    # with open(feminas_dir + '/Runs/'+model+'/V.py', 'w') as f:
    #   #f.write("""fmodel_name='%s'\n"""% model_name)
    #   #f.write("""ffeminas_dir = '%s' """ % feminas_dir)
    #   f.write('import numpy as np \n')
    #   f.write('\n')
    #   for i in femL:
    #     f.write("""%s='%s'\n"""% (i,eval(i)))
    #   for i in readL:
    #     f.write("""%s=%s\n"""% (i,eval(i)))
    #   for i in topL:
    #     f.write("""%s=%s\n"""% (i,eval(i)))
    #   for i in boundaryL:
    #     f.write("""%s=%s\n"""% (i,eval(i)))
    #   for i in constantL:
    #     f.write("""%s=%s\n"""% (i,eval(i)))
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_config
    write_config(locals())

  print('Running Variables')
