import numpy as np
#import pyNastran
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import pdb
import os

mesh=BDF(debug=True,log=None)

# Beam Segments structure: 
#========================
NumBeams=1
class Structure():     
  pass

global BeamSeg
BeamSeg=[Structure() for i in range(NumBeams)]


# Geometry
#=============================================================

L=[10.]
N=[51]
W=[1. for i in range(NumBeams)]
H=[1. for i in range(NumBeams)]
TH=[0.1 for i in range(NumBeams)]
E=[7.e10 for i in range(NumBeams)]
J=[3.e-8 for i in range(NumBeams)]
Area=[np.pi*0.014**2 for i in range(NumBeams)]
NU=[0.33 for i in range(NumBeams)]
I1=[1./7*1e-3 for i in range(NumBeams)]
I2=[1./7*1e-8 for i in range(NumBeams)]
Direc=np.asarray([[1,0,0]])
PID=[1 for i in range(NumBeams)]
rho=2.703e3

for i in range(NumBeams):

 BeamSeg[i].Lx=L[i]
 BeamSeg[i].nx=N[i]
 BeamSeg[i].w=W[i]
 BeamSeg[i].h=H[i]
 BeamSeg[i].th=TH[i]
 BeamSeg[i].nu=NU[i]
 BeamSeg[i].w=W[i]
 BeamSeg[i].e=E[i]
 BeamSeg[i].j=J[i]
 BeamSeg[i].area=Area[i]
 BeamSeg[i].I1=I1[i]
 BeamSeg[i].I2=I2[i]
 BeamSeg[i].direc=Direc[i]/np.linalg.norm(Direc[i])
 if i==0:
  BeamSeg[i].dl=BeamSeg[i].Lx/(BeamSeg[i].nx-1)
 else:
  BeamSeg[i].dl=BeamSeg[i].Lx/(BeamSeg[i].nx)
 BeamSeg[i].idmat=i+1
 BeamSeg[i].idpbeam=i+1
 BeamSeg[i].pid=PID[i]
 BeamSeg[i].NodeX=np.zeros((BeamSeg[i].nx,3))

BeamConn=[0]
Node0=np.array([0.,0.,0.])

for i in range(NumBeams):
    for j in range(BeamSeg[i].nx):
      
       if i==0:
         if j==0:
           x0=Node0
           BeamSeg[i].NodeX[j] = x0 
         else:  
           x0=BeamSeg[i].NodeX[j-1] 
           BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl 
       else:
         if j==0:
           x0=BeamSeg[BeamConn[i]].NodeX[-1]
           BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl 
         #elif j==BeamSeg[i].nx-1:
         #  continue
         else:
           x0=BeamSeg[i].NodeX[j-1]    
           BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl 
           
       

NumNodes=sum(N)
#Mass
#Mass=224*np.ones(na)
#I=np.zeros((6,na))
#for i in range(na):
#  I[0,i]=37.36;I[1,i]=0.;I[2,i]=18.68;I[3,i]=0.0;I[4,i]=0.0;I[5,i]=18.68
 #I11        I21       I22   g     I31        I32        I33



#=============================================================================================================
#===== Executive control =====================================================================================

sol='103'


#========================================================
# Sol 400
#========================================================

if sol=='400':
  mesh.executive_control_lines = ['SOL '+sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  spc=2
  numloads=5
  #mesh.Set=1,
  #mesh._write_sets
  case_control = 'TITLE=Beam1_%s \nECHO=NONE \n'%(sol)
  for i in range(numloads):

    ccx='SUBCASE %s \n  STEP 1 \n  SUBTITLE=load%s \n  ANALYSIS = NLSTATIC \n  NLSTEP = %s \n  SPC = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) =ALL \n' %(i+1,i+1,i+1,spc,2*(i+1)+1)

    case_control=case_control + ccx
  case_control=case_control + 'BEGIN BULK \nPARAM,LGDISP,1 \nPARAM,POST,-1 \n'
  for i in range(numloads):
    ccy='NLSTEP,%s \n' %(i+1)
    case_control=case_control + ccy

  mesh.case_control_deck = case_control


#========================================================
# Sol 103
#========================================================

elif sol=='103':
  pch=1
  mesh.executive_control_lines = ['SOL '+sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  method=1
  spc=2

  if pch:
    case_control = 'TITLE=Beam1_%s \nECHO=NONE  \nSUBCASE 1 \nSPC=%s  \nBEGIN BULK \nPARAM,EXTOUT,DMIGPCH \n' %(sol,spc)
  else:
    case_control = 'TITLE=Beam1_%s \nECHO=NONE  \nSUBCASE 1 \nMETHOD=%s \n  SPC=%s \nVECTOR(SORT1,REAL)=ALL \nBEGIN BULK \nPARAM,POST,-2 \n$PARAM,EXTOUT,DMIGPCH \n' %(sol,method,spc)

  mesh.case_control_deck = case_control


# Eigenvector Card
#==========================
  if pch==0:
    SID = method
    ND = NumNodes*6
    eigrl = ['EIGRL', SID, None, None, ND, None, None, None,'MASS']
    mesh.add_card(eigrl, 'EIGRL')


#========================================================
# Sol 101
#========================================================

elif sol=='101':

  sol='101'
  mesh.executive_control_lines = ['SOL '+sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  spc=2
  numloads=5
  #mesh.Set=1,2
  #mesh._write_sets
  case_control = 'TITLE=Wing1_%s \nECHO=NONE  \n'%(sol)
  for i in range(numloads):

    ccx='SUBCASE %s  \n  SUBTITLE=load%s \n  ANALYSIS = STATIC  \n  SPC = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) = ALL \n' %(i+1,i+1,spc,2*(i+1)+1)
    case_control=case_control + ccx

  case_control=case_control + 'BEGIN BULK \nPARAM,POST,-1 \n'
  mesh.case_control_deck = case_control


#======================================================================================================================================================
#=============Bulk Data=================================================================================================================================


# Material cards
#===================================================================================================================================== 

for i in range(NumBeams):
  Em=BeamSeg[i].e
  Nu=BeamSeg[i].nu
  id_mat=BeamSeg[i].idmat
  mat1 = ['MAT1',id_mat,Em,None,Nu,None]
  mesh.add_card(mat1,'MAT1')

  



# Beam properties flat
#=====================================================================================================================================
pbeaml=0
if pbeaml:
  for i in range(NumBeams):
    Em=BeamSeg[i].e
    Nu=BeamSeg[i].nu
    id_mat=BeamSeg[i].idmat
    id_p=BeamSeg[i].idpbeam
    w=BeamSeg[i].w
    h=BeamSeg[i].h
    th1=BeamSeg[i].th
    th2=BeamSeg[i].th
    pbeam = ['PBEAML',id_p,id_mat,Aa,I1a,I2a,I12a,Ja,None]
    mesh.add_card(pbeam,'PBEAML')
else:

    for i in range(NumBeams):
      Em=BeamSeg[i].e
      Nu=BeamSeg[i].nu
      id_mat=BeamSeg[i].idmat
      id_p=BeamSeg[i].idpbeam
      
      Aa=BeamSeg[i].area
      I1a=BeamSeg[i].I1
      I2a=BeamSeg[i].I2
      I12a=None
      #Ja=I1a+I2a
      Ja=BeamSeg[i].j
      #pbeam = ['PBEAM',id_p,id_mat,Aa,I1a,I2a,I12a,Ja]
      pbeam = ['PBEAM',id_p,id_mat,Aa,I1a,I2a,I12a,Ja]
      mesh.add_card(pbeam,'PBEAM')


    

# Nodes
#=====================================================================================================================================


# idx=0
# for i in range(NumBeams):
#     PID=BeamSeg[i].pid
#     X1=0. ; X2=1.; X3=0.
#     for k in range(BeamSeg[i].nx):

#         x=k*BeamSeg[i].direc[0]*BeamSeg[i].dl
#         y=k*BeamSeg[i].direc[1]*BeamSeg[i].dl
#         z=k*BeamSeg[i].direc[2]*BeamSeg[i].dl
#         Id=(k+1)+idx
#         node=['GRID',Id,None,x,y,z,None,None,None]    
#         mesh.add_card(node,'GRID')
        
#         if k==BeamSeg[i].nx-1:
#          continue

#         EID=Id 
#         GA=Id ;GB=Id+1
#         cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
#         mesh.add_card(cbeam,'CBEAM')

#     idx=idx+BeamSeg[i].nx    

#pdb.set_trace()
idx=0
G2X={}
X2G={}

for i in range(NumBeams):
    
    
    for k in range(BeamSeg[i].nx):

        x=BeamSeg[i].NodeX[k,0]
        y=BeamSeg[i].NodeX[k,1]
        z=BeamSeg[i].NodeX[k,2]
        Id=(k+1)+idx
        G2X[Id]=BeamSeg[i].NodeX[k]
        X2G[tuple(BeamSeg[i].NodeX[k])]=Id
        node=['GRID',Id,None,x,y,z,None,None,None]    
        mesh.add_card(node,'GRID')
        
                 

        
    idx=idx+BeamSeg[i].nx    

EID=0
for i in range(NumBeams):
    PID=BeamSeg[i].idpbeam
    X1=0. ; X2=1.; X3=0.
    for k in range(BeamSeg[i].nx-1):
        
        
        EID=EID+1
        if i>0 and k==0:
            GA=X2G[tuple(BeamSeg[BeamConn[i]].NodeX[-1])];GB=X2G[tuple(BeamSeg[i].NodeX[k])]
            cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
            mesh.add_card(cbeam,'CBEAM')
            EID=EID+1
        
        GA=X2G[tuple(BeamSeg[i].NodeX[k])] ;GB=X2G[tuple(BeamSeg[i].NodeX[k+1])]
        cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
        mesh.add_card(cbeam,'CBEAM')

    idx=idx+BeamSeg[i].nx    
    

#====================================================================================================================================


# Masses 
#=============================================================================================================
conm2true=1
if conm2true:
  mass=rho*Area[0]*L[0]/N[0]
  I11=mass*J[0]
  I22=mass*I1[0]
  I33=mass*I2[0]
  I21=I31=I32=0.
  X1=0.0;X2=0.0;X3=0.0
  idm=0
  for i in range(NumBeams):
    for k in range(BeamSeg[i].nx):

          Eid= idx+idm+(k) 
          RefGid=(k+1)+idm

          CONM2=['CONM2',Eid,RefGid,0,mass,X1,X2,X3,None,I11,I21,I22,I31,I32,I33]
          mesh.add_card(CONM2,'CONM2')            

    idm=idm+BeamSeg[i].nx

else:
  idm=0
  w=BeamSeg[i].w
  h=BeamSeg[i].h
  th1=BeamSeg[i].th
  th2=BeamSeg[i].th
  I1a=2*w*th1*(h/2)**2+2*(1./12)*th1*h**3
  I2a=2*h*th1*(w/2)**2+2*(1./12)*th1*w**3
  Aa=2*h*th1+2*w*th1
  m11=rho*Aa; m22=rho*Aa; m33 = rho*Aa; m44=rho*(I1a+I2a);
  m21=m31=m32=m41=m42=m43=m51=m52=m53=m54=m55=m61=m62=m63=m64=m65=m66=0.
  for i in range(NumBeams):
    for k in range(BeamSeg[i].nx):

          Eid= idx+idm+(k) 
          RefGid=(k+1)+idm

          CONM1=['CONM1',Eid,RefGid,0,m11,m21,m31,m32,m33, m41,m42,m43,m44,m51,m52,m53,m54,m55,m61,m62,m63,m64,m65,m66]
          mesh.add_card(CONM1,'CONM1')            

    idm=idm+BeamSeg[i].nx  
#=============================================================================================================



# Fix constraints
#===========================================================================================================

#mesh.add_card_fields(['dd',1,3],'dd')
#mesh.add_spcadd('[SPCADD,2,1])',SPCADD)

Sid=1
#mesh.add_spcadd(spc,[Sid])
mesh.add_card(['SPCADD',spc,Sid],'SPCADD')

C=123456
clamped_node=1
spc1=['SPC1',Sid,C,clamped_node]
mesh.add_card(spc1,'SPC1')

# Load and Force Cards
#===========================================================================================================


#pdb.set_trace()
if sol is not '103':

  
  numForce=1
  Fl=[[np.pi*1,np.pi*5,np.pi*10,np.pi*15,np.pi*20]]
  Fd=[[0.,1.,0.]]
  GF=[X2G[tuple(BeamSeg[0].NodeX[-1])]]
  for i in range(numloads):
    sid=2*(i+1)+1
    load1=['LOAD',sid,1.]
    for j in range(numForce):

     sid=2*(i+1)+j*10
     G= GF[j]
     F= Fl[j][i]
     force1=['MOMENT',sid,G,0,F,Fd[j][0],Fd[j][1],Fd[j][2]]
     mesh.add_card(force1,'MOMENT')
     load1.append(1.)
     load1.append(2*(i+1)+j*10)
    mesh.add_card(load1,'LOAD')




# Output file
#===========================================================================================================

folder0='Beam'+sol

folder1=''
for i in range(NumBeams):

  folder1=folder1+'Lx%s' % int(BeamSeg[i].Lx)

folder2=''
for i in range(NumBeams):
  folder2=folder2+'nx%s'%(BeamSeg[i].nx)

if not os.path.exists(folder0+'/'+folder1+'/'+folder2):
  os.makedirs(folder0+'/'+folder1+'/'+folder2)


x0=np.array([1,0,0])
femname='S_'
for i in range(NumBeams):
  femname=femname+'%s' % int(np.arccos(BeamSeg[i].direc.dot(x0)))

   
mesh.write_bdf(folder0+'/'+folder1+'/'+folder2+'/'+femname+ '.bdf')
#===========================================================================================================
