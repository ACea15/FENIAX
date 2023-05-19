from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import pdb
import os

torun='Shell_Rafa'

import importlib
inp=importlib.import_module('Generators'+'.'+'WingGen'+'.'+torun+'.'+'Wing_Inp')
reload(inp)

mesh=BDF(debug=True,log=None)

if inp.rotation:
    import intrinsic.Tools.transformations
    Rba=intrinsic.Tools.transformations.rotation_matrix(inp.rot_angle,inp.rot_direc)[:3, :3]

# Tapper functions
#===================================================================================================================================
def f(x):

    Ly=inp.Lyr-(inp.Lyr-inp.Lyt)/inp.Lx*x
    return Ly


def g(x):

    Lz=inp.Lzr-(inp.Lzr-inp.Lzt)/inp.Lx*x
    return Lz


# Aset Nodes
#=============================================================================================================
aset1=['ASET1',123456]

for i in range(inp.na):

        Id= 500000 + int(round(inp.aset[i]/inp.dlx))
        aset1.append(Id)

# Aset Card
#==========================
if not inp.dynamic:
    mesh.add_card(aset1,'ASET1')


count=1
set1=''
for e in aset1[2:]:
  if count<len(aset1[2:]):
   comma=','
  else:
   comma=''
  if count%8==0:
   set1=set1+'\n'
   set1=set1+str(e)+comma
  else:
   set1= set1+str(e)+comma
  count=count+1



setn=1

#=============================================================================================================
#===== Executive control =====================================================================================

#========================================================
# Sol 400
#========================================================

# if inp.sol=='400':
#   mesh.executive_control_lines = ['SOL '+inp.sol,'CEND']
#   #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
#   spc=2
#   #numloads=5
#   #mesh.Set=1,2
#   #mesh._write_sets
#   case_control = 'TITLE=Wing1_%s \nECHO=NONE \nSET %s = %s \n'%(inp.sol,setn,set1)
#   for i in range(inp.numLoads):

#     ccx='SUBCASE %s \n  STEP 1 \n  SUBTITLE=load%s \n  ANALYSIS = NLSTATIC \n  NLSTEP = %s \n  SPC = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) = %s \n' %(i+1,i+1,i+1,spc,2*(i+1)+1,setn)

#     case_control=case_control + ccx
#   case_control=case_control + 'BEGIN BULK \nPARAM,LGDISP,1 \nPARAM,POST,-1 \n'
#   for i in range(inp.numloads):
#     ccy='NLSTEP,%s \n' %(i+1)
#     case_control=case_control + ccy

#   mesh.case_control_deck = case_control

if inp.sol=='400':


 if inp.static:
  mesh.executive_control_lines = ['SOL '+inp.sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  spc=2

  #mesh.Set=1,
  #mesh._write_sets
  case_control = 'TITLE=Wing1_%s \nECHO=NONE \nSET %s = %s \n'%(inp.sol,setn,set1)
  for i in range(inp.numLoads):

    ccx='SUBCASE %s \n  STEP 1 \n  SUBTITLE=load%s \n  ANALYSIS = NLSTATIC \n  NLSTEP = %s \n  SPC = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) = %s \n' %(i+1,i+1,i+1,spc,2*(i+1)+1,setn)

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
  case_control = 'TITLE=Wing1_%s \nECHO=NONE \nSET %s = %s \n'%(inp.sol,setn,set1)
  if inp.Velocity0 or inp.Displacement0 :
    for i in range(1):

      ccx='SUBCASE %s \n  SUBTITLE=load%s \n  ANALYSIS = NLTRAN \n  NLSTEP = %s \n  SPC = %s \n  IC=%s  \n  DISPLACEMENT(SORT1,REAL) = %s \n' %(i+1,i+1,i+1,spc,i+1,setn)

      case_control=case_control + ccx
    case_control=case_control + 'BEGIN BULK \nPARAM,LGDISP,1 \nPARAM,POST,-1 \n'
    #for i in range(inp.numLoads):
    #  ccy='NLSTEP,%s \n' %(i+1)
    #  case_control=case_control + ccy


    mesh.case_control_deck = case_control
    none6=[None for i in range(6)]
    none7=[None for i in range(7)]
    nlstep=['NLSTEP',1,inp.tf]+none6+["GENERAL"]+none7+["FIXED",inp.tni]
    #nlstep=['NLSTEP',1,]
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
# Sol 103
#========================================================

elif inp.sol=='103':
  #pch=0
  mesh.executive_control_lines = ['SOL '+inp.sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  method=1
  spc=2

  if inp.free:
      if inp.pch:
        case_control = 'TITLE=Wing1_%s \nECHO=NONE \nSET %s = %s  \nSUBCASE 1 \n  SUBTITLE=na_%s  \nBEGIN BULK \nPARAM,EXTOUT,DMIGPCH \n' %(inp.sol,setn,set1,inp.na)
      else:
        case_control = 'TITLE=Wing1_%s \nECHO=NONE \nSET %s = %s  \nSUBCASE 1 \n  SUBTITLE=na_%s \n  METHOD=%s \n  VECTOR(SORT1,REAL)=%s \nBEGIN BULK \nPARAM,POST,-2 \n$PARAM,EXTOUT,DMIGPCH \n' %(inp.sol,setn,set1,inp.na,method,setn)

  else:
      if inp.pch:
        case_control = 'TITLE=Wing1_%s \nECHO=NONE \nSET %s = %s  \nSUBCASE 1 \n  SUBTITLE=na_%s \n  SPC=%s  \nBEGIN BULK \nPARAM,EXTOUT,DMIGPCH \n' %(inp.sol,setn,set1,inp.na,spc)
      else:
        case_control = 'TITLE=Wing1_%s \nECHO=NONE \nSET %s = %s  \nSUBCASE 1 \n  SUBTITLE=na_%s \n  METHOD=%s \n  SPC=%s \n  VECTOR(SORT1,REAL)=%s \nBEGIN BULK \nPARAM,POST,-2 \n$PARAM,EXTOUT,DMIGPCH \n' %(inp.sol,setn,set1,inp.na,method,spc,setn)

  mesh.case_control_deck = case_control


# Eigenvector Card
#==========================
  if inp.pch==0:
    SID = method
    ND = inp.na*6
    eigrl = ['EIGRL', SID, None, None, ND, None, None, None,'MASS']
    mesh.add_card(eigrl, 'EIGRL')



#========================================================
# Sol 101
#========================================================

elif inp.sol=='101':

  #sol='101'
  mesh.executive_control_lines = ['SOL '+inp.sol,'CEND']
  #mesh.case_control_deck = 'TITLE=SAILPLANE \nECHO=NONE \nSUBCASE 1 \nMETHOD=1 \nSPC=1 \nVECTOR=ALL \nWEIGHTCHECK(PRINT,SET=ALL)=YES \nPARAM,POST,-1\nBEGIN BULK\n'
  spc=2
  #numloads=5
  #mesh.Set=1,2
  #mesh._write_sets
  case_control = 'TITLE=Wing1_%s \nECHO=NONE \nSET %s = %s \n'%(inp.sol,setn,set1)
  for i in range(inp.numloads):

    ccx='SUBCASE %s  \n  SUBTITLE=load%s \n  ANALYSIS = STATIC  \n  SPC = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) = %s \n' %(i+1,i+1,spc,2*(i+1)+1,setn)

    case_control=case_control + ccx

  case_control=case_control + 'BEGIN BULK \nPARAM,POST,-1 \n'
  mesh.case_control_deck = case_control




#======================================================================================================================================================
#=============Bulk Data=================================================================================================================================


# Material cards
#=====================================================================================================================================
if inp.lumped:
    id_mat=1; mat1 = ['MAT1',id_mat,inp.Em,None,inp.Nu,None]
    mesh.add_card(mat1,'MAT1')
else:
    id_mat=1; mat1 = ['MAT1',id_mat,inp.Em,None,inp.Nu,inp.rho]
    mesh.add_card(mat1,'MAT1')


# Shell properties flat
#=====================================================================================================================================
id_p=1; pshell1 = ['PSHELL',id_p,id_mat,inp.thickness,id_mat,None,id_mat]
mesh.add_card(pshell1,'PSHELL')

# Nodes
#=====================================================================================================================================
for i in range(inp.nx):
    for j in range(inp.ny-1):

        x=i*inp.dlx
        y=f(x)/2 - f(x)/inp.Lyr*inp.dly*j + inp.tipy/inp.Lyr*x
        z=g(x)/2+inp.tipz/inp.Lzr*x
        if inp.rotation:
            x,y,z=Rba.dot([x,y,z])
        Id=100000+1000*j+i
        node=['GRID',Id,None,x,y,z,None,None,None]
        mesh.add_card(node,'GRID')

        x=i*inp.dlx
        y=-f(x)/2 + f(x)/inp.Lyr*inp.dly*j + inp.tipy/inp.Lyr*x
        z=-g(x)/2+inp.tipz/inp.Lzr*x
        if inp.rotation:
            x,y,z=Rba.dot([x,y,z])
        Id=300000+1000*j+i
        node=['GRID',Id,None,x,y,z,None,None,None]
        mesh.add_card(node,'GRID')

for i in range(inp.nx):
    for k in range(inp.nz-1):

        x=i*inp.dlx
        y=-f(x)/2+inp.tipy/inp.Lyr*x
        z= g(x)/2-g(x)/inp.Lzr*inp.dlz*k+ inp.tipz/inp.Lzr*x
        if inp.rotation:
            x,y,z=Rba.dot([x,y,z])
        Id=200000+1000*k+i
        node=['GRID',Id,None,x,y,z,None,None,None]
        mesh.add_card(node,'GRID')

        x=i*inp.dlx
        y=f(x)/2+inp.tipy/inp.Lyr*x
        z= -g(x)/2+g(x)/inp.Lzr*inp.dlz*k+ inp.tipz/inp.Lzr*x
        if inp.rotation:
            x,y,z=Rba.dot([x,y,z])
        Id=400000+1000*k+i
        node=['GRID',Id,None,x,y,z,None,None,None]
        mesh.add_card(node,'GRID')

AsetId=[]
for i in range(inp.na):

        x = inp.aset[i]
        y = inp.tipy/inp.Lyr*x
        z = inp.tipz/inp.Lzr*x
        if inp.rotation:
            x,y,z=Rba.dot([x,y,z])
        Id= 500000 + int(round(inp.aset[i]/inp.dlx))
        node=['GRID',Id,None,x,y,z,None,None,None]
        AsetId.append(Id)
        mesh.add_card(node,'GRID')

#====================================================================================================================================



#Elements
#====================================================================================================================================
h=1
for i in range(inp.nx-1):
    for j in range(inp.ny-1):

        if j==inp.ny-2:
            Id=h
            G1=100000+1000*j+i
            G2=200000+1000*(0)+i
            G3=200000+1000*(0)+(i+1)
            G4=100000+1000*j+(i+1)
            quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
            mesh.add_card(quad,'CQUAD4')
            h=h+1
        else:
            Id=h
            G1=100000+1000*j+i
            G2=100000+1000*(j+1)+i
            G3=100000+1000*(j+1)+(i+1)
            G4=100000+1000*j+(i+1)
            quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
            mesh.add_card(quad,'CQUAD4')
            h=h+1


        if j==inp.ny-2:
            Id=h
            G1=300000+1000*(inp.ny-2)+i
            G2=400000+1000*(0)+i
            G3=400000+1000*(0)+(i+1)
            G4=300000+1000*(inp.ny-2)+(i+1)
            quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
            mesh.add_card(quad,'CQUAD4')
            h=h+1
        else:
            Id=h
            G1=300000+1000*j+i
            G2=300000+1000*(j+1)+i
            G3=300000+1000*(j+1)+(i+1)
            G4=300000+1000*j+(i+1)
            quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
            mesh.add_card(quad,'CQUAD4')
            h=h+1


for i in range(inp.nx-1):
    for k in range(inp.nz-1):

        if k==inp.nz-2:
            Id=h
            G1=200000+1000*k+i
            G2=300000+1000*(0)+i
            G3=300000+1000*(0)+(i+1)
            G4=200000+1000*k+(i+1)
            quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
            mesh.add_card(quad,'CQUAD4')
            h=h+1
        else:
            Id=h
            G1=200000+1000*k+i
            G2=200000+1000*(k+1)+i
            G3=200000+1000*(k+1)+(i+1)
            G4=200000+1000*k+(i+1)
            quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
            mesh.add_card(quad,'CQUAD4')
            h=h+1
        if k==inp.nz-2:
            Id=h
            G1=400000+1000*(inp.nz-2)+i
            G2=100000+1000*(0)+i
            G3=100000+1000*(0)+(i+1)
            G4=400000+1000*(inp.nz-2)+(i+1)
            quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
            mesh.add_card(quad,'CQUAD4')
            h=h+1
        else:
            Id=h
            G1=400000+1000*k+i
            G2=400000+1000*(k+1)+i
            G3=400000+1000*(k+1)+(i+1)
            G4=400000+1000*k+(i+1)
            quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
            mesh.add_card(quad,'CQUAD4')
            h=h+1
#=============================================================================================================


# Masses
#=============================================================================================================
# for i in range(inp.na):

#         Eid= 600000 + int(round(inp.aset[i]/inp.dlx))
#         RefGid= 500000 + int(round(inp.aset[i]/inp.dlx))
#         mass=inp.Mass[i]
#         I11=inp.I[0,i];I21=inp.I[1,i];I22=inp.I[2,i];I31=inp.I[3,i];I32=inp.I[4,i];I33=inp.I[5,i]
#         X1=0.0;X2=0.0;X3=0.0
#         CONM2=['CONM2',Eid,RefGid,0,mass,X1,X2,X3,None,I11,I21,I22,I31,I32,I33]
#         mesh.add_card(CONM2,'CONM2')
if inp.lumped:
    for i in range(inp.na):

            Eid= 600000 + int(round(inp.aset[i]/inp.dlx))
            RefGid= 500000 + int(round(inp.aset[i]/inp.dlx))
            mass=inp.Mass[i]
            I11=inp.I[0,i];I21=inp.I[1,i];I22=inp.I[2,i];I31=inp.I[3,i];I32=inp.I[4,i];I33=inp.I[5,i]
            X1=0.0;X2=0.0;X3=0.0
            CONM2=['CONM2',Eid,RefGid,0,mass,X1,X2,X3,None,I11,I21,I22,I31,I32,I33]
            mesh.add_card(CONM2,'CONM2')


#=============================================================================================================

# RBE3s
#=============================================================================================================
for i in range(inp.na):

    Eid = 700000 + int(round(inp.aset[i]/inp.dlx))
    RefGid = 500000 + int(round(inp.aset[i]/inp.dlx))
    W1=1.
    RefC=123456
    C1=123
    RBE3 = ['RBE3',Eid,None,RefGid,RefC,W1,C1]

    for j in range(inp.ny-1):
     Gi1 = 100000+1000*j+int(round(inp.aset[i]/inp.dlx))
     RBE3.append(Gi1)
     Gi2 = 300000+1000*j+int(round(inp.aset[i]/inp.dlx))
     RBE3.append(Gi2)

    for k in range(inp.nz-1):
     Gi3 = 200000+1000*k+int(round(inp.aset[i]/inp.dlx))
     RBE3.append(Gi3)
     Gi4 = 400000+1000*k+int(round(inp.aset[i]/inp.dlx))
     RBE3.append(Gi4)

    Gm1=100000+1000*0+int(round(inp.aset[i]/inp.dlx))
    Cm1=123
    Gm2=300000+1000*0+int(round(inp.aset[i]/inp.dlx))
    Cm2=12
    Gm3=200000+1000*0+int(round(inp.aset[i]/inp.dlx))
    Cm3=1
    RBE3.append('UM')
    RBE3.append(Gm1)
    RBE3.append(Cm1)
    RBE3.append(Gm2)
    RBE3.append(Cm2)
    RBE3.append(Gm3)
    RBE3.append(Cm3)
    mesh.add_card(RBE3,'RBE3')


# Fix constraints
#===========================================================================================================

#mesh.add_card_fields(['dd',1,3],'dd')
#mesh.add_spcadd('[SPCADD,2,1])',SPCADD)
if not inp.free:
    Sid=1
    #mesh.add_spcadd(spc,[Sid])
    mesh.add_card(['SPCADD',spc,Sid],'SPCADD')

    C=123456
    spc1=['SPC1',Sid,C]
    for j in range(inp.ny-1):
     Gi1 = 100000+1000*j
     spc1.append(Gi1)
     Gi2= 300000+1000*j
     spc1.append(Gi2)

    for k in range(inp.nz-1):
     Gi3 = 200000+1000*k
     spc1.append(Gi3)
     Gi4 = 400000+1000*k
     spc1.append(Gi4)

    mesh.add_card(spc1,'SPC1')


# Load and Force Cards
#===========================================================================================================
if inp.sol is not '103':

  # for i in range(inp.numloads):

  #   sid=2*(i+1)
  #   G=aset1[-1]
  #   F=5.*10**(i+2)
  #   force1=['FORCE',sid,G,0,F,0.0,0.0,-1.]
  #   mesh.add_card(force1,'FORCE')

  #   sid=2*(i+1)+1
  #   load1=['LOAD',sid,1.,1.,2*(i+1)]
  #   mesh.add_card(load1,'LOAD')


  if inp.FORCE1:
    for i in range(inp.numLoads):
      lid=2*(i+1)+1
      load1=['LOAD',lid,1.]
      for j in range(inp.numForce):

       if inp.Fl is not None:
         gridpoint = mesh.nodes[aset1[inp.gridF[i][j]]].get_position()+inp.Fd[i][j]
         gridId = (j+1)*1000+i
         node=['GRID',gridId,None,gridpoint[0],gridpoint[1],gridpoint[2],None,None,None]
         mesh.add_card(node,'GRID')

         sid=2*(i+1)+(j+1)*100
         G=aset1[inp.gridF[i][j]]
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
         gridpointm = mesh.nodes[aset1[inp.gridF[i][j]]].get_position()+inp.Md[i][j]
         if np.allclose(gridpoint,gridpointm) is not True:
           gridId = (j+1)*1110+i
           node=['GRID',gridId,None,gridpointm[0],gridpointm[1],gridpointm[2],None,None,None]
           mesh.add_card(node,'GRID')
           G=aset1[inp.gridF[i][j]]
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

# Initial conditions
#===========================

if inp.sol=='400' and inp.dynamic and inp.Velocity0:

    for vi in range(len(inp.vg)):
      Id = AsetId[vi]
      GId = mesh.Node(Id).get_position()
      #print GId
      for d in range(3):

          tic=['TIC',1,Id,d+1,0.,inp.fv(GId)[d]]
          mesh.add_card(tic,'TIC')


if inp.sol=='400' and inp.dynamic  and inp.Displacement0:

    for ui in range(len(inp.ug)):
      Id =  AsetId[ui]
      GId = mesh.Node(Id).get_position()
      for d in range(3):

          tic=['TIC',1,Id,d+1,inp.fu(GId)[d]]
          mesh.add_card(tic,'TIC')

if inp.sol=='400' and inp.dynamic  and inp.Acceleration0:

    for ai in range(len(inp.ag)):
      Id = mesh.asets[0].node_ids[ai]
      GId = mesh.Node(Id).get_position()
      for d in range(3):

          tic=['TIC',1,Id,d+1,inp.fa(GId)[d]]
          mesh.add_card(tic,'TIC')


# Output file
#===========================================================================================================
fdis='/nx%sny%snz%s' %(inp.nx,inp.ny,inp.nz)
if not os.path.exists(inp.feminas_dir+'/Models'+'/'+inp.model+'/Wing'+inp.sol+fdis):
  os.makedirs( inp.feminas_dir+'/Models'+'/'+inp.model+'/Wing'+inp.sol+fdis)

mesh.write_bdf( inp.feminas_dir+'/Models'+'/'+inp.model+'/Wing'+inp.sol+fdis+'/'+inp.femName)



#folder0='BeamWing'+inp.sol

#folder1='BW_Lx%sLyr%sLyt%sLzr%sLzt%sty%stz%s' %(int(Lx),int(Lyr),int(Lyt),int(Lzr),int(Lzt),int(round(tipy)),int(round(tipz)))
#folder2='nx%sny%snz%s' %(nx,ny,nz)
#if not os.path.exists(folder0+'/'+folder1+'/'+folder2):
#  os.makedirs(folder0+'/'+folder1+'/'+folder2)

#femName = 'S_na%s.bdf' %(inp.na)
#mesh.write_bdf(folder0+'/'+folder1+'/'+folder2+'/'+femName)
#===========================================================================================================
