

import numpy as np
import os
#import sys

Grid='structuralGrid'
K_a='Kaa.npy'
M_a='Maa.npy'

model_name='/'+'/'.join(os.getcwd().split('/')[-2:])
model=os.getcwd().split('/')[-1]
for i in range(len(os.getcwd().split('/'))):
  if os.getcwd().split('/')[-i-1]=='PyFem2NL2':
    pyfem2nl_dir="/".join(os.getcwd().split('/')[0:-i])

 
Grid = pyfem2nl_dir + model_name+'/FEM/'+Grid
K_a = pyfem2nl_dir +model_name+'/FEM/'+K_a
M_a = pyfem2nl_dir +model_name+'/FEM/'+M_a


NumModes = 150#100#6*25#35
NumBeams=1
Nastran_modes=0
if Nastran_modes:
 op2name=['S_0']
else:
 op2name = ''


# Loading
if 'loads.py' in os.listdir(os.getcwd()):
  loading=1
else:
  loading=0
 
# Reading Grid file
#============================================
node_start=1              # NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start # Number Nodes 
start_reading=3     #range(start_reading,len(lin)):
beam_start=1           # j=int(s[4])-beam_start 
nodeorder_start=500000    # aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start) 

RigidBody_Modes=0
Clamped=1
ClampX=[]
BeamsClamped=[]
if Clamped:
  BeamsClamped=[0]
  ClampX='np.array([0.,0.,0.])'
  
#[[i] for i in range(1,50) ]
#BeamConn = [[[1],[2],[3],[4],[5],[6],[7],[8],[9],[]],[[],[0],[1],[2],[3],[4],[5],[6],[7],[8]]]

BeamConn = [[[]],[[]]]

EMAT='np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])'  #EMAT matrix
I3='np.eye(3)'
e_1='np.array([1,0,0])'


# ======================================================================================================
# Time Discretization
#======================================================================================================
t0=0;tf=1;tn=11

if (__name__ == '__main__'):

  saving=1
  if saving:
    femL=['Grid','K_a','M_a','op2name','pyfem2nl_dir','model_name','model']
    topL=['NumModes','NumBeams','BeamConn','Nastran_modes','loading','t0','tf','tn']
    readL=['node_start','start_reading','beam_start','nodeorder_start']
    boundaryL=['RigidBody_Modes','Clamped','ClampX','BeamsClamped']
    constantL=['EMAT','I3','e_1']

    if not os.path.exists(pyfem2nl_dir + '/Runs/'+model):
      os.makedirs(pyfem2nl_dir + '/Runs/'+model)
      with open(pyfem2nl_dir + '/Runs/'+model+'/__init__.py', 'w') as f2:
        print('Init file created')

    with open(pyfem2nl_dir + '/Runs/'+model+'/V.py', 'w') as f:
      #f.write("""fmodel_name='%s'\n"""% model_name)
      #f.write("""fpyfem2nl_dir = '%s' """ % pyfem2nl_dir)
      f.write('import numpy as np \n')
      f.write('\n')
      for i in femL:
        f.write("""%s='%s'\n"""% (i,eval(i)))
      for i in readL:
        f.write("""%s=%s\n"""% (i,eval(i)))
      for i in topL:
        f.write("""%s=%s\n"""% (i,eval(i)))
      for i in boundaryL:
        f.write("""%s=%s\n"""% (i,eval(i)))
      for i in constantL:
        f.write("""%s=%s\n"""% (i,eval(i)))


  print('Running Variables')


    


