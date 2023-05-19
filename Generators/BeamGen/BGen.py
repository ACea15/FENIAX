import numpy as np
#import pyNastran
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import pdb
import os

mesh=BDF(debug=True,log=None)

import importlib
from Run import torun
inp=importlib.import_module("Generators"+'.'+'BeamGen'+'.'+torun+'.'+'BGen_Inp')
reload(inp)
# Beam Segmexonts structure:
#========================

class Structure():
  pass

global BeamSeg
BeamSeg=[Structure() for i in range(inp.NumBeams)]


for i in range(inp.NumBeams):

 BeamSeg[i].Lx=inp.L[i]
 BeamSeg[i].nx=inp.N[i]
 try:
   BeamSeg[i].w=inp.W[i]
   BeamSeg[i].h=inp.H[i]
   BeamSeg[i].w=inp.W[i]
   BeamSeg[i].th=inp.TH[i]
 except AttributeError:
   printx

 BeamSeg[i].nu=inp.NU[i]
 BeamSeg[i].e=inp.E[i]
 BeamSeg[i].j=inp.J[i]
 BeamSeg[i].area=inp.Area[i]
 BeamSeg[i].I1=inp.I1[i]
 BeamSeg[i].I2=inp.I2[i]
 BeamSeg[i].K1=inp.K1[i]
 BeamSeg[i].K2=inp.K2[i]
 BeamSeg[i].direc=inp.Direc[i]/np.linalg.norm(inp.Direc[i])
 if inp.Clamped and i in inp.BeamsClamped:
  BeamSeg[i].dl=BeamSeg[i].Lx/(BeamSeg[i].nx-1)
 elif inp.Clamped==0 and i==0:
  BeamSeg[i].dl=BeamSeg[i].Lx/(BeamSeg[i].nx-1)
 else:
  BeamSeg[i].dl=BeamSeg[i].Lx/(BeamSeg[i].nx)

 BeamSeg[i].idmat=i+1
 BeamSeg[i].idpbeam=i+1
 BeamSeg[i].pid=inp.PID[i]
 BeamSeg[i].NodeX=np.zeros((BeamSeg[i].nx,3))
 BeamSeg[i].rho=inp.rho[i]



if inp.Clamped:
  for i in range(inp.NumBeams):
      for j in range(BeamSeg[i].nx):

         if i in inp.BeamsClamped:
           if j==0:
             x0=inp.Node0
             BeamSeg[i].NodeX[j] = x0
           else:
             x0=BeamSeg[i].NodeX[j-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl
         else:
           if j==0:
             x0=BeamSeg[inp.BeamConn[i]].NodeX[-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl
           #elif j==BeamSeg[i].nx-1:
           #  continue
           else:
             x0=BeamSeg[i].NodeX[j-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl

else:
  for i in range(inp.NumBeams):
      for j in range(BeamSeg[i].nx):

         if i==0:
           if j==0:
             x0=inp.Node0
             BeamSeg[i].NodeX[j] = x0
           else:
             x0=BeamSeg[i].NodeX[j-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl
         else:
           if j==0:
             x0=BeamSeg[inp.BeamConn[i]].NodeX[-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl
           #elif j==BeamSeg[i].nx-1:
           #  continue
           else:
             x0=BeamSeg[i].NodeX[j-1]
             BeamSeg[i].NodeX[j] = x0 + BeamSeg[i].direc*BeamSeg[i].dl


NumNodes=sum(inp.N)


#Mass
#Mass=224*np.ones(na)
#I=np.zeros((6,na))
#for i in range(na):
#  I[0,i]=37.36;I[1,i]=0.;I[2,i]=18.68;I[3,i]=0.0;I[4,i]=0.0;I[5,i]=18.68
 #I11        I21       I22   g     I31        I32        I33



#=============================================================================================================
#===== Executive control =====================================================================================

#========================================================
# Inp.Sol 400
#========================================================

if inp.sol=='400':


 if inp.static:
  mesh.executive_control_lines = ['SOL '+inp.sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  spc=2

  #mesh.Set=1,
  #mesh._write_sets
  case_control = 'TITLE=Beam1_%s \nECHO=NONE \n'%(inp.sol)
  for i in range(inp.numLoads):

    ccx='SUBCASE %s \n  STEP 1 \n  SUBTITLE=load%s \n  ANALYSIS = NLSTATIC \n  NLSTEP = %s \n  SPC = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) =ALL \n' %(i+1,i+1,i+1,spc,2*(i+1)+1)

    case_control=case_control + ccx
  case_control=case_control + 'BEGIN BULK \nPARAM,LGDISP,1 \nPARAM,POST,-1 \n'
  for i in range(inp.numLoads):
    ccy='NLSTEP,%s \n' %(i+1)
    case_control=case_control + ccy

  mesh.case_control_deck = case_control


 if inp.dynamic:

  mesh.executive_control_lines = ['SOL '+inp.sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  spc=2

  #mesh.Set=1,
  #mesh._write_sets
  case_control = 'TITLE=Beam1_%s \nECHO=NONE \n'%(inp.sol)
  if inp.Velocity0:
    #for i in range(inp.numLoads):

    ccx='SUBCASE %s \n  SUBTITLE=load%s \n  ANALYSIS = NLTRAN \n  NLSTEP = %s \n  SPC = %s \n  IC=%s  \n  DISPLACEMENT(SORT1,REAL) =ALL \n' %(i+1,i+1,i+1,spc,i+1)

    case_control=case_control + ccx
    case_control=case_control + 'BEGIN BULK \nPARAM,LGDISP,1 \nPARAM,POST,-1 \n'
    #for i in range(inp.numLoads):
    #  ccy='NLSTEP,%s \n' %(i+1)
    #  case_control=case_control + ccy


    mesh.case_control_deck = case_control
    none6=[None for i in range(6)]
    none7=[None for i in range(7)]
    nlstep=['NLSTEP',1,20.]+none6+["GENERAL"]+none7+["FIXED",inp.ti]
    mesh.add_card(nlstep, 'NLSTEP')

  else:

    for i in range(inp.numLoads):

      ccx='SUBCASE %s \n  SUBTITLE=load%s \n  ANALYSIS = NLTRAN \n  NLSTEP = %s \n  SPC = %s  \n  DISPLACEMENT(SORT1,REAL) =ALL \n' %(i+1,i+1,i+1,spc)

      case_control=case_control + ccx
    case_control=case_control + 'BEGIN BULK \nPARAM,LGDISP,1 \nPARAM,POST,-1 \n'
  #for i in range(inp.numLoads):
  #  ccy='NLSTEP,%s \n' %(i+1)
  #  case_control=case_control + ccy

  mesh.case_control_deck = case_control




#========================================================
# Inp.Sol 103
#========================================================

elif inp.sol=='103':

  mesh.executive_control_lines = ['SOL '+inp.sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  method=1
  spc=2

  if inp.pch:
    case_control = 'TITLE=Beam1_%s \nECHO=NONE  \nSUBCASE 1 \nSPC=%s  \nBEGIN BULK \nPARAM,EXTOUT,DMIGPCH \n' %(inp.sol,spc)
  else:
    case_control = 'TITLE=Beam1_%s \nECHO=NONE  \nSUBCASE 1 \nMETHOD=%s \n  SPC=%s \nVECTOR(SORT1,REAL)=ALL \nBEGIN BULK \nPARAM,POST,-2 \n$PARAM,EXTOUT,DMIGPCH \n' %(inp.sol,method,spc)

  mesh.case_control_deck = case_control

#=====================================================================





# Eigenvector Card
#==========================
  if inp.pch==0:
    SID = method
    ND = NumNodes*6
    eigrl = ['EIGRL', SID, None, None, ND, None, None, None,'MASS']
    mesh.add_card(eigrl, 'EIGRL')


#========================================================
# Inp.Sol 101
#========================================================

elif inp.sol=='101':

  inp.sol='101'
  mesh.executive_control_lines = ['SOL '+inp.sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  spc=2
  #mesh.Set=1,2
  #mesh._write_sets
  case_control = 'TITLE=Wing1_%s \nECHO=NONE  \n'%(inp.sol)
  for i in range(inp.numLoads):

    ccx='SUBCASE %s  \n  SUBTITLE=load%s \n  ANALYSIS = STATIC  \n  SPC = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) = ALL \n' %(i+1,i+1,spc,2*(i+1)+1)
    case_control=case_control + ccx

  case_control=case_control + 'BEGIN BULK \nPARAM,POST,-1 \n'
  mesh.case_control_deck = case_control


#======================================================================================================================================================
#=============Bulk Data=================================================================================================================================


# Material cards
#=====================================================================================================================================

for i in range(inp.NumBeams):
  Em=BeamSeg[i].e
  Nu=BeamSeg[i].nu
  rho1=BeamSeg[i].rho
  id_mat=BeamSeg[i].idmat
  if inp.density:
    mat1 = ['MAT1',id_mat,Em,None,Nu,rho1]
  else:
    mat1 = ['MAT1',id_mat,Em,None,Nu,None]
  mesh.add_card(mat1,'MAT1')





# Beam properties flat
#=====================================================================================================================================
pbeaml=0
if pbeaml:
  for i in range(inp.NumBeams):
    Em=BeamSeg[i].e
    Nu=BeamSeg[i].nu
    id_mat=BeamSeg[i].idmat
    id_p=BeamSeg[i].idpbeam
    w=BeamSeg[i].w
    h=BeamSeg[i].h
    th1=BeamSeg[i].th
    th2=BeamSeg[i].th
    pbeam = ['PBEAML',id_p,id_mat,None,'Bar',None,None,None,None,w,h]
    mesh.add_card(pbeam,'PBEAML')
else:

    for i in range(inp.NumBeams):
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
      if BeamSeg[i].K1 is not None:
       n08 = [None for ix in range(8)];n07 = [None for ix in range(7)]
       pbeam = pbeam +[None] + n08 + ['NO'] + [1.] + n07 + n07 \
               +[BeamSeg[i].K1,BeamSeg[i].K2]
       #print pbeam

      mesh.add_card(pbeam,'PBEAM')




# Nodes
#=====================================================================================================================================

# idx=0
# for i in range(inp.NumBeams):
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

for i in range(inp.NumBeams):


    for k in range(BeamSeg[i].nx):

        x=BeamSeg[i].NodeX[k,0]
        y=BeamSeg[i].NodeX[k,1]
        z=BeamSeg[i].NodeX[k,2]
        Id=(k+1)+idx
        G2X[Id]=BeamSeg[i].NodeX[k]
        X2G[tuple(BeamSeg[i].NodeX[k])]=Id
        node=['GRID',Id,None,x,y,z,None,None,None]
        mesh.add_card(node,'GRID')

        if inp.sol=='400' and k!=0 and inp.Velocity0:
          tic=['TIC',1,Id,1,0.,inp.fv([x,y,z])[0]]
          mesh.add_card(tic,'TIC')
          tic=['TIC',1,Id,2,0.,inp.fv([x,y,z])[1]]
          mesh.add_card(tic,'TIC')
          tic=['TIC',1,Id,3,0.,inp.fv([x,y,z])[2]]
          mesh.add_card(tic,'TIC')



    idx=idx+BeamSeg[i].nx
#pdb.set_trace()
EID=1
for i in range(inp.NumBeams):
    PID=BeamSeg[i].idpbeam
    X1=0. ; X2=0.; X3=1.

    if BeamSeg[i].nx == 1:

        #EID=EID+1
        #pdb.set_trace()
        GA=X2G[tuple(BeamSeg[inp.BeamConn[i]].NodeX[-1])];GB=X2G[tuple(BeamSeg[i].NodeX[0])]
        cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
        mesh.add_card(cbeam,'CBEAM')
        EID=EID+1
    else:
      #pdb.set_trace()
      for k in range(BeamSeg[i].nx):

          #EID=EID+1

        if inp.Clamped:
          if  i in inp.BeamsClamped and k!=BeamSeg[i].nx-1:
              GA=X2G[tuple(BeamSeg[i].NodeX[k])] ;GB=X2G[tuple(BeamSeg[i].NodeX[k+1])]
              cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
              mesh.add_card(cbeam,'CBEAM')
              EID=EID+1
          elif  i in inp.BeamsClamped and k==BeamSeg[i].nx-1:
              continue

          elif   i not in inp.BeamsClamped and i!=0 and k==0:
              GA=X2G[tuple(BeamSeg[inp.BeamConn[i]].NodeX[-1])];GB=X2G[tuple(BeamSeg[i].NodeX[k])]
              cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
              mesh.add_card(cbeam,'CBEAM')
              EID=EID+1
          else:
              GA=X2G[tuple(BeamSeg[i].NodeX[k-1])] ;GB=X2G[tuple(BeamSeg[i].NodeX[k])]
              cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
              mesh.add_card(cbeam,'CBEAM')
              EID=EID+1

        else:
          if  i==0 and k!=BeamSeg[i].nx-1:
              GA=X2G[tuple(BeamSeg[i].NodeX[k])] ;GB=X2G[tuple(BeamSeg[i].NodeX[k+1])]
              cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
              mesh.add_card(cbeam,'CBEAM')
              EID=EID+1
          elif  i==0 and k==BeamSeg[i].nx-1:
              continue

          elif  i!=0 and k==0:
              GA=X2G[tuple(BeamSeg[inp.BeamConn[i]].NodeX[-1])];GB=X2G[tuple(BeamSeg[i].NodeX[k])]
              cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
              mesh.add_card(cbeam,'CBEAM')
              EID=EID+1
          else:
              GA=X2G[tuple(BeamSeg[i].NodeX[k-1])] ;GB=X2G[tuple(BeamSeg[i].NodeX[k])]
              cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
              mesh.add_card(cbeam,'CBEAM')
              EID=EID+1




    idx=idx+BeamSeg[i].nx


#====================================================================================================================================


# Masses
#=============================================================================================================

if inp.conm1:
    idm=0
    for i in range(inp.NumBeams):
      for k in range(BeamSeg[i].nx):

            Eid= idx+idm+(k)
            RefGid=(k+1)+idm

            CONM1=['CONM1',Eid,RefGid,0,inp.m11[i][k],inp.m21[i][k],inp.m22[i][k],
                   inp.m31[i][k],inp.m32[i][k],inp.m33[i][k], inp.m41[i][k],
                   inp.m42[i][k],inp.m43[i][k],inp.m44[i][k],inp.m51[i][k],
                   inp.m52[i][k],inp.m53[i][k],inp.m54[i][k],inp.m55[i][k],
                   inp.m61[i][k],inp.m62[i][k],inp.m63[i][k],inp.m64[i][k],
                   inp.m65[i][k],inp.m66[i][k]]
            mesh.add_card(CONM1,'CONM1')

      idm=idm+BeamSeg[i].nx

if inp.conm2:


    idm=0
    for i in range(inp.NumBeams):
      for k in range(BeamSeg[i].nx):

            Eid= idx+idm+(k)
            RefGid=(k+1)+idm

            CONM2=['CONM2',Eid,RefGid,0,inp.mass[i][k],
                   inp.X1[i][k],inp.X2[i][k],inp.X3[i][k],None,
                   inp.I11[i][k],inp.I21[i][k], inp.I22[i][k],
                   inp.I31[i][k],inp.I32[i][k],inp.I33[i][k]]
            mesh.add_card(CONM2,'CONM2')

      idm=idm+BeamSeg[i].nx


old_mass=0
if old_mass:
  conm2true=1
  if conm2true:


    idm=0
    for i in range(inp.NumBeams):
      if i in inp.BeamsClamped:
        mass=BeamSeg[i].rho*BeamSeg[i].area*BeamSeg[i].Lx/(BeamSeg[i].nx-1)
      else:
        mass=BeamSeg[i].rho*BeamSeg[i].area*BeamSeg[i].Lx/BeamSeg[i].nx
      I11=mass*BeamSeg[i].j
      I22=mass*BeamSeg[i].I1
      I33=mass*BeamSeg[i].I2
      I21=I31=I32=0.


      X1=0.0;X2=0.0;X3=0.0
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

    for i in range(inp.NumBeams):
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
if inp.Clamped:
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
if inp.sol is not '103':

  FORCE=0
  if FORCE:
    for i in range(inp.numLoads):
      sid=2*(i+1)+1
      load1=['LOAD',sid,1.]
      for j in range(inp.numForce):

       sid=2*(i+1)+j*10
       G= X2G[tuple(BeamSeg[inp.gridF[i][j][0]].NodeX[inp.gridF[i][j][1]])]
       F= inp.Fl[i][j]
       force1=['FORCE',sid,G,0,F,inp.Fd[i][j][0],inp.Fd[i][j][1],inp.Fd[i][j][2]]
       mesh.add_card(force1,'FORCE')
       load1.append(1.)
       load1.append(2*(i+1)+j*10)
      mesh.add_card(load1,'LOAD')

  FORCE1=1
  if FORCE1:
    for i in range(inp.numLoads):
      lid=2*(i+1)+1
      load1=['LOAD',lid,1.]
      for j in range(inp.numForce):

       if inp.Fl is not None:
         gridpoint = BeamSeg[inp.gridF[i][j][0]].NodeX[inp.gridF[i][j][1]]+inp.Fd[i][j]
         gridId = (j+1)*1000+i
         node=['GRID',gridId,None,gridpoint[0],gridpoint[1],gridpoint[2],None,None,None]
         mesh.add_card(node,'GRID')

         sid=2*(i+1)+(j+1)*100
         G= X2G[tuple(BeamSeg[inp.gridF[i][j][0]].NodeX[inp.gridF[i][j][1]])]
         F= inp.Fl[i][j]
         G1=G
         G2=gridId

         rbe2=['RBE2',(j+1)*100+i,G,123456,G2]
         mesh.add_card(rbe2,'RBE2')

         force1=['FORCE1',sid,G,F,G1,G2]
         mesh.add_card(force1,'FORCE1')
         load1.append(1.)
         load1.append(sid)

       if inp.Ml is not None:
         gridpointm = BeamSeg[inp.gridF[i][j][0]].NodeX[inp.gridF[i][j][1]]+inp.Md[i][j]
         if np.allclose(gridpoint,gridpointm) is not True:
           gridId = (j+1)*1110+i
           node=['GRID',gridId,None,gridpointm[0],gridpointm[1],gridpointm[2],None,None,None]
           mesh.add_card(node,'GRID')
           G= X2G[tuple(BeamSeg[inp.gridF[i][j][0]].NodeX[inp.gridF[i][j][1]])]
           G1=G
           G2=gridId

           rbe2=['RBE2',(j+1)*100+i,G,123456,G2]
           mesh.add_card(rbe2,'RBE2')

         sid=2*(i+1)+(j+1)*100+1
         #print sid
         M = inp.Ml[i][j]
         moment1=['MOMENT1',sid,G,M,G1,G2]
         mesh.add_card(moment1,'MOMENT1')
         load1.append(1.)
         load1.append(sid)
      mesh.add_card(load1,'LOAD')


  PRESSURE=0
  if PRESSURE:

    for i in range(inp.numLoads):
      sid=2*(i+1)+1
      load1=['LOAD',sid,1.]
      for j in range(inp.N[0]-1):

       sid2=2000*(i+1)+j*10
       load1.append(1.)
       load1.append(sid2)
       pl1=['PLOAD1',sid2,j+1,'FZE','FR',0.,inp.Fp[i],1.,inp.Fp[i]]
       mesh.add_card(pl1,'PLOAD1')


      mesh.add_card(load1,'LOAD')



# Output file
#===========================================================================================================


if not os.path.exists(inp.feminas_dir+'/Models'+'/'+inp.model+'/Beam'+inp.sol):
  os.makedirs( inp.feminas_dir+'/Models'+'/'+inp.model+'/Beam'+inp.sol)


mesh.write_bdf( inp.feminas_dir+'/Models'+'/'+inp.model+'/Beam'+inp.sol+'/'+inp.femname)


'''
folder0='Beam'+inp.sol

folder1=''
for i in range(inp.NumBeams):

  folder1=folder1+'Lx%s' % int(BeamSeg[i].Lx)

folder2=''
for i in range(inp.NumBeams):
  folder2=folder2+'nx%s'%(BeamSeg[i].nx)

if not os.path.exists(folder0+'/'+folder1+'/'+folder2):
  os.makedirs(folder0+'/'+folder1+'/'+folder2)


x0=np.array([1,0,0])
femname='S_'
for i in range(inp.NumBeams):
  femname=femname+'%s' % int(np.arccos(BeamSeg[i].direc.dot(x0)))


mesh.write_bdf(folder0+'/'+folder1+'/'+folder2+'/'+femname+ '.bdf')
'''
#===========================================================================================================


def dispx():
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for i in range(inp.NumBeams):

      x = BeamSeg[i].NodeX[:,0]
      y = BeamSeg[i].NodeX[:,1]
      z = BeamSeg[i].NodeX[:,2]

      #ax.scatter(x, y, z, c='r', marker='o')
      ax.plot(x, y, z, c='r', marker='o')


    #plt.axis('off')

    #fig.suptitle('Mode'+str(modeplot)+'_Phi1:'+str(r))
    #plt.axis([0,80,-1,1,-3,3])
  plt.show()
