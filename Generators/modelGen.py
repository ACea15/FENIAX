import numpy as np
import pdb
import os
import importlib
import subprocess

class Model:

    def __init__(self,inp,mesh):
        self.mesh = mesh
        self.inp = importlib.import_module(inp)
        self.path = self.inp.feminas_dir+'/Models'+'/'+ self.inp.model+'/NASTRAN'
    def aelist(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_aelist(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'AELIST')
    def aset1(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_aset1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'ASET1')
    def aefact(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_aefact(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'AEFACT')
    def aero(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_aero(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'AERO')
    def caero1(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_caero1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'CAERO1')
    def cbeam(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_cbeam(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'CBEAM')
    def cquad4(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_cbeam(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'CQUAD4')
    def conm1(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_conm1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'CONM1')
    def conm2(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_conm2(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'CONM2')
    def dload(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_dload(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'DLOAD')
    def eigr(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_eigr(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'EIGR')
    def eigrl(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_eigrl(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'EIGRL')
    def grids(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_grid(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'GRID')
    def force(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_force(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'FORCE')
    def force1(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_force1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'FORCE1')
    def load(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_load(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'LOAD')
    def mat1(self,var):
        if type(var).__name__ == 'dict':
          self.mesh.add_mat1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'MAT1')
    def moment1(self,var):
        if type(var).__name__ == 'dict':
          self.mesh.add_mat1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'MOMENT1')
    def param(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_param(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'PARAM')
    def paero1(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_paero1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'PAERO1')
    def pbeam(self,var,pbeaml=0):
        if pbeaml:
            if type(var).__name__ == 'dict':
                self.mesh.add_pbeaml(**var)
            if type(var).__name__ == 'list':
                self.mesh.add_card(var,'PBEAML')
        else:
            if type(var).__name__ == 'dict':
                self.mesh.add_pbeam(**var)
            if type(var).__name__ == 'list':
                self.mesh.add_card(var,'PBEAM')
    def pshell(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_cbeam(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'PSHELL')
    def rbe2(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_cbeam(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'RBE2')
    def rbe3(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_cbeam(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'RBE3')
    def set1(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_set1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'SET1')
    def spc(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_spc(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'SPC1')
    def spcadd(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_spcadd(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'SPCADD')
    def spline(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_spline(**var) #! Needs defining a spline number
        if type(var).__name__ == 'list':
            self.mesh.add_card(var[1:],'SPLINE%s'%var[0])
    def tabled1(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_tabled1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'TABLED1')
    def tload1(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_tload1(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'TLOAD1')            
    def tic(self,var):
        if type(var).__name__ == 'dict':
            self.mesh.add_cbeam(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'TIC')
    def nlstep(self,var):
        if type(var).__name__== 'dict':
            self.mesh.add_spcadd(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'NLSTEP')
    def tstep(self,var):
        if type(var).__name__== 'dict':
            self.mesh.add_spcadd(**var)
        if type(var).__name__ == 'list':
            self.mesh.add_card(var,'TSTEP')

    def executive_control_deck(self,ecd=''):
        #===== Executive control ================================
        if ecd:
            self.mesh.executive_control_lines = ecd
        else:
            self.mesh.executive_control_lines = ['SOL '+self.inp.sol,'CEND']

    def case_control_deck(self,ccd='',add_ccd=0,setx=[]):
        # The Case Control Section performs the following basic functions:
        # Selects loads and constraints.
        # Defines the contents of the Model Results Output File.
        # Defines the output coordinate system for element and grid point results.
        # Defines the subcase structure for the analysis.

        case_control = 'TITLE=%s_%s \nECHO=NONE \n'%(self.inp.model,self.inp.sol)
        # for i in self.inp.spc:
        #     case_control += 'SPC=%s  \n' % i
        case_control += 'SPC=%s  \n' % 2
        if setx: 
            count=1
            set1=''
            for e in setx:
              if count<len(setx):
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
            case_control +="SET %s = %s \n"%(setn,set1)
        else:
            setn = 'ALL'
            set1=''
        def ccd103(case_control):
          #========================================================
          # Sol 103
          #========================================================
          method=1
          if self.inp.pch:
            case_control += 'SUBCASE 1 \nBEGIN BULK \nPARAM,EXTOUT,DMIGPCH \n'
          else:
            case_control += 'SUBCASE 1 \nMETHOD=%s \nVECTOR(SORT1,REAL)=%s \nBEGIN BULK \nPARAM,POST,-2 \n$PARAM,EXTOUT,DMIGPCH \n' %(method,setn)
          return case_control
        def ccd101(case_control):
          #========================================================
          # Inp.Sol 101
          #========================================================
          for i in range(inp.numLoads):

            ccx='SUBCASE %s  \n  SUBTITLE=load%s \n  ANALYSIS = STATIC  \n  SPC = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) = %s \n' %(i+1,i+1,spc,2*(i+1)+1,setn)
            case_control=case_control + ccx

          case_control=case_control + 'BEGIN BULK \nPARAM,POST,-1 \n'
          return case_control
        def ccd400(case_control):
          #========================================================
          # Sol 400
          #========================================================
          for i in range(self.inp.numLoads):

            ccx0='SUBCASE %s \n  SUBTITLE=load%s \n' %(i+1,i+1)
            if self.inp.static:
              ccx1='  ANALYSIS = NLSTATIC \n'
              ccx2='  NLSTEP = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) =%s \n' %(i+1,2*(i+1)+1,setn)
            if self.inp.dynamic:
              ccx1='  ANALYSIS = NLTRAN \n'
              if self.inp.Velocity0:
               ccx1+='  IC=%s'%(i+1)
               ccx2='  NLSTEP = %s \n  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) =%s \n' %(i+1,2*(i+1)+1,setn)
              else:
               ccx2='  NLSTEP = %s \n  DLOAD = %s  \n  DISPLACEMENT(SORT1,REAL) =%s \n' %(i+1,7*(i+1)+1,setn)
            case_control += ccx0+ccx1+ccx2

          case_control += 'BEGIN BULK \nPARAM,LGDISP,1 \nPARAM,POST,-1 \n'
          for i in range(self.inp.numLoads):
            if self.inp.static:
                ccx2='NLSTEP, %s \n' %(i+1)
                case_control += ccx2

          return case_control

        def ccd109(case_control):
          #========================================================
          # Sol 109
          #========================================================
          for i in range(self.inp.numLoads):

            ccx0='SUBCASE %s \n  SUBTITLE=load%s \n' %(i+1,i+1)
            ccx1=''
            if self.inp.Velocity0:
               ccx1+='  IC=%s'%(i+1)
               ccx2='  LOAD = %s  \n  DISPLACEMENT(SORT1,REAL) =%s \n' %(2*(i+1)+1,setn)
            else:
               ccx2='  DLOAD = %s  \n  DISPLACEMENT(SORT1,REAL) =%s \n' %(7*(i+1)+1,setn)
            case_control += ccx0+ccx1+ccx2
            case_control += 'BEGIN BULK \nPARAM,POST,-1 \n'

          return case_control
     
        if add_ccd:
            self.mesh.case_control_deck = case_control+ccd+'BEGIN BULK\n'
        elif ccd:
            self.mesh.case_control_deck = ccd
        else:

            if self.inp.sol=='400':

              self.mesh.case_control_deck = ccd400(case_control)


            elif self.inp.sol=='109':

              self.mesh.case_control_deck = ccd109(case_control)


            elif self.inp.sol=='103':

              self.mesh.case_control_deck = ccd103(case_control)

            elif self.inp.sol=='101':
              self.mesh.case_control_deck = ccd101(case_control)

    def write_model(self,model='/m1.bdf',size=8,is_double=False,add_string=None):

        if not os.path.exists(self.path):
          os.makedirs(self.path)
        self.mesh.write_bdf(self.path+model, encoding=None, size=size, is_double=is_double,
                            interspersed=False, enddata=None, close=True)
        if add_string is not None:
            for si in add_string.keys():
                if type(add_string[si]).__name__ == 'str':
                    self.write_string(string_find=add_string[si],string_write=si,line_find=None,model=model)
                if type(add_string[si]).__name__ == 'int':
                    self.write_string(string_find=None,string_write=si,line_find=add_string[si],model=model)

    def write_includes(self,includes,model='/m1.bdf'):
        with open(self.path+model,'a') as Mbdf:
            for i in includes:
                Mbdf.write("INCLUDE '%s'" %i + '\n')
    def write_string(self,string_find,string_write,line_find=None,model='/m1.bdf'):

        with open(self.path+model,'r') as file1:

            lines = file1.readlines()
            if line_find:
                lw = line_find
            else:
                for li in range(len(lines)):
                    if lines[li] == string_find:
                        lw = li+1
            linesw = lines[:lw]+[string_write]+lines[lw:]

        with open(self.path+model,'w') as file1:

            file1.writelines(linesw)


    def write_modes(self,torun,NumModes,rb=1,variables='',cwd=''):

        import Runs.Torun
        from pyNastran.op4.op4 import OP4
        op4 = OP4()
        #pdb.set_trace()
        if variables:
            subprocess.call("python " + variables,shell=True,cwd=cwd)
        else:
            subprocess.call(["python",self.inp.feminas_dir+'/Models'+'/'+ self.inp.model+'/variables.py'],cwd=self.inp.feminas_dir+'/Models'+'/'+ self.inp.model)
        Runs.Torun.torun = torun #'GolandWinglong'
        if rb:
            import intrinsic.modesrb
            from intrinsic.modesrb import Phi0V
        else:
            import intrinsic.modes
            from intrinsic.modes import Phi0V
        self.Phi0 = np.array(Phi0V[:NumModes]).T
        #print np.shape(self.Phi0)
        #pdb.set_trace()
        Phi1_dic={}
        Phi1_dic['Phi1']=(2,self.Phi0)
        Phi0_dic={}
        Phi0_dic['Phi0']=(2,self.Phi0)
        op4.write_op4(self.path+'/Phi1.op4',Phi1_dic,is_binary=False)
        op4.write_op4(self.path+'/Phi0.op4',Phi0_dic,is_binary=False)


class Beam_model:
    def __init__(self,inp,ins):
        self.inp = importlib.import_module(inp)
        self.ins = ins
    # Beam Segmexonts structure:
    #========================

        class Structure():
          pass

        self.BeamSeg=[Structure() for i in range(self.inp.NumBeams)]

        for i in range(self.inp.NumBeams):

         self.BeamSeg[i].Lx=self.inp.L[i]
         self.BeamSeg[i].nx=self.inp.N[i]
         try:
           self.BeamSeg[i].w=self.inp.W[i]
           self.BeamSeg[i].h=self.inp.H[i]
           self.BeamSeg[i].w=self.inp.W[i]
           self.BeamSeg[i].th=self.inp.TH[i]
         except AttributeError:
           print

         self.BeamSeg[i].nu=self.inp.NU[i]
         self.BeamSeg[i].X=self.inp.beamcross[i]
         self.BeamSeg[i].e=self.inp.E[i]
         self.BeamSeg[i].j=self.inp.J[i]
         self.BeamSeg[i].area=self.inp.Area[i]
         self.BeamSeg[i].I1=self.inp.I1[i]
         self.BeamSeg[i].I2=self.inp.I2[i]
         self.BeamSeg[i].K1=self.inp.K1[i]
         self.BeamSeg[i].K2=self.inp.K2[i]
         self.BeamSeg[i].direc=self.inp.Direc[i]/np.linalg.norm(self.inp.Direc[i])
         if self.inp.Clamped and i in self.inp.BeamsClamped:
          self.BeamSeg[i].dl=self.BeamSeg[i].Lx/(self.BeamSeg[i].nx-1)
         elif self.inp.Clamped==0 and i==0:
          self.BeamSeg[i].dl=self.BeamSeg[i].Lx/(self.BeamSeg[i].nx-1)
         else:
          self.BeamSeg[i].dl=self.BeamSeg[i].Lx/(self.BeamSeg[i].nx)

         self.BeamSeg[i].idmat=i+1
         self.BeamSeg[i].idpbeam=i+1
         self.BeamSeg[i].pid=self.inp.PID[i]
         self.BeamSeg[i].NodeX=np.zeros((self.BeamSeg[i].nx,3))
         self.BeamSeg[i].rho=self.inp.rho[i]



        if self.inp.Clamped:
          for i in range(self.inp.NumBeams):
              for j in range(self.BeamSeg[i].nx):

                 if i in self.inp.BeamsClamped:
                   if j==0:
                     x0=self.inp.Node0
                     self.BeamSeg[i].NodeX[j] = x0
                   else:
                     x0=self.BeamSeg[i].NodeX[j-1]
                     self.BeamSeg[i].NodeX[j] = x0 + self.BeamSeg[i].direc*self.BeamSeg[i].dl
                 else:
                   if j==0:
                     x0=self.BeamSeg[self.inp.BeamConn[i]].NodeX[-1]
                     self.BeamSeg[i].NodeX[j] = x0 + self.BeamSeg[i].direc*self.BeamSeg[i].dl
                   #elif j==self.BeamSeg[i].nx-1:
                   #  continue
                   else:
                     x0=self.BeamSeg[i].NodeX[j-1]
                     self.BeamSeg[i].NodeX[j] = x0 + self.BeamSeg[i].direc*self.BeamSeg[i].dl

        else:
          for i in range(self.inp.NumBeams):
              for j in range(self.BeamSeg[i].nx):

                 if i==0:
                   if j==0:
                     x0=self.inp.Node0
                     self.BeamSeg[i].NodeX[j] = x0
                   else:
                     x0=self.BeamSeg[i].NodeX[j-1]
                     self.BeamSeg[i].NodeX[j] = x0 + self.BeamSeg[i].direc*self.BeamSeg[i].dl
                 else:
                   if j==0:
                     x0=self.BeamSeg[self.inp.BeamConn[i]].NodeX[-1]
                     self.BeamSeg[i].NodeX[j] = x0 + self.BeamSeg[i].direc*self.BeamSeg[i].dl
                   #elif j==self.BeamSeg[i].nx-1:
                   #  continue
                   else:
                     x0=self.BeamSeg[i].NodeX[j-1]
                     self.BeamSeg[i].NodeX[j] = x0 + self.BeamSeg[i].direc*self.BeamSeg[i].dl


        NumNodes=sum(self.inp.N)

    def beams_initials(self,ND=50):
        # Eigenvector Card
        #==========================
        if inp.pch==0:
          SID = method
          #ND = NumNodes*6
          eigrl = ['EIGRL', SID, None, None, ND, None, None, None,'MASS']
          self.ins.eigrl(eigrl)

    def beams(self):

        def beams_materials(self):

            # Material cards
            #=====================================================
            for i in range(self.inp.NumBeams):
              Em=self.BeamSeg[i].e
              Nu=self.BeamSeg[i].nu
              rho1=self.BeamSeg[i].rho
              id_mat=self.BeamSeg[i].idmat
              if 0:#self.inp.density:
                mat1 = ['MAT1',id_mat,Em,None,Nu,rho1]
              else:
                mat1 = ['MAT1',id_mat,Em,None,Nu,None]
              self.ins.mat1(mat1)

        def beams_properties(self):
            # Beam properties flat
            #=====================================================
            if 0:#self.inp.pbeaml:
                for i in range(self.inp.NumBeams):
                  Em=self.BeamSeg[i].e
                  Nu=self.BeamSeg[i].nu
                  id_mat=self.BeamSeg[i].idmat
                  id_p=self.BeamSeg[i].idpbeam
                  w=self.BeamSeg[i].w
                  h=self.BeamSeg[i].h
                  th1=self.BeamSeg[i].th
                  th2=self.BeamSeg[i].th
                  pbeam = ['PBEAML',id_p,id_mat,None,'Bar',None,None,None,None,w,h]
                  self.ins.pbeam(pbeam,1)
            else:
                for i in range(self.inp.NumBeams):
                  Em=self.BeamSeg[i].e
                  Nu=self.BeamSeg[i].nu
                  id_mat=self.BeamSeg[i].idmat
                  id_p=self.BeamSeg[i].idpbeam
                  Aa=self.BeamSeg[i].area
                  I1a=self.BeamSeg[i].I1
                  I2a=self.BeamSeg[i].I2
                  I12a=None
                  #Ja=I1a+I2a
                  Ja=self.BeamSeg[i].j
                  #pbeam = ['PBEAM',id_p,id_mat,Aa,I1a,I2a,I12a,Ja]
                  pbeam = ['PBEAM',id_p,id_mat,Aa,I1a,I2a,I12a,Ja]
                  if self.BeamSeg[i].K1 is not None:
                   n08 = [None for ix in range(8)];n07 = [None for ix in range(7)]
                   pbeam = pbeam +[None] + n08 + ['NO'] + [1.] + n07 + n07 \
                           +[self.BeamSeg[i].K1,self.BeamSeg[i].K2]
                  self.ins.pbeam(pbeam)

        def beams_grids(self):
            self.idx=0
            self.G2X={}
            self.X2G={}

            for i in range(self.inp.NumBeams):
                for k in range(self.BeamSeg[i].nx):

                    x=self.BeamSeg[i].NodeX[k,0]
                    y=self.BeamSeg[i].NodeX[k,1]
                    z=self.BeamSeg[i].NodeX[k,2]
                    Id=(k+1)+self.idx
                    self.G2X[Id]=self.BeamSeg[i].NodeX[k]
                    self.X2G[tuple(self.BeamSeg[i].NodeX[k])]=Id
                    node=['GRID',Id,None,x,y,z,None,None,None]
                    self.ins.grids(node)
                    # if self.inp.sol=='400' and k!=0 and self.inp.Velocity0:
                    #   tic=['TIC',1,Id,1,0.,self.inp.fv([x,y,z])[0]]
                    #   mesh.add_card(tic,'TIC')
                    #   tic=['TIC',1,Id,2,0.,self.inp.fv([x,y,z])[1]]
                    #   mesh.add_card(tic,'TIC')
                    #   tic=['TIC',1,Id,3,0.,self.inp.fv([x,y,z])[2]]
                    #   mesh.add_card(tic,'TIC')

                self.idx=self.idx+self.BeamSeg[i].nx
            if self.inp.Velocity0:
                none6=[None for i in range(6)]
                none7=[None for i in range(7)]
                nlstep=['NLSTEP',1,20.]+none6+["GENERAL"]+none7+["FIXED",self.inp.ti]
                self.ins.nlstep(nlstep)
        def beams_elements(self):
            EID=1
            for i in range(self.inp.NumBeams):
                PID=self.BeamSeg[i].idpbeam
                X1=self.BeamSeg[i].X[0] ; X2=self.BeamSeg[i].X[1]; X3=self.BeamSeg[i].X[2]

                if self.BeamSeg[i].nx == 1:

                    #EID=EID+1
                    GA=self.X2G[tuple(self.BeamSeg[self.inp.BeamConn[i]].NodeX[-1])];GB=self.X2G[tuple(self.BeamSeg[i].NodeX[0])]
                    cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
                    self.ins.cbeam(cbeam)
                    EID=EID+1
                else:
                  #pdb.set_trace()
                  for k in range(self.BeamSeg[i].nx):

                      #EID=EID+1

                    if self.inp.Clamped:
                      if  i in self.inp.BeamsClamped and k!=self.BeamSeg[i].nx-1:
                          GA=self.X2G[tuple(self.BeamSeg[i].NodeX[k])] ;GB=self.X2G[tuple(self.BeamSeg[i].NodeX[k+1])]
                          cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
                          self.ins.cbeam(cbeam)
                          EID=EID+1
                      elif  i in self.inp.BeamsClamped and k==self.BeamSeg[i].nx-1:
                          continue

                      elif   i not in self.inp.BeamsClamped and i!=0 and k==0:
                          GA=self.X2G[tuple(self.BeamSeg[self.inp.BeamConn[i]].NodeX[-1])];GB=self.X2G[tuple(self.BeamSeg[i].NodeX[k])]
                          cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
                          self.ins.cbeam(cbeam)
                          EID=EID+1
                      else:
                          GA=self.X2G[tuple(self.BeamSeg[i].NodeX[k-1])] ;GB=self.X2G[tuple(self.BeamSeg[i].NodeX[k])]
                          cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
                          self.ins.cbeam(cbeam)
                          EID=EID+1

                    else:
                      if  i==0 and k!=self.BeamSeg[i].nx-1:
                          GA=self.X2G[tuple(self.BeamSeg[i].NodeX[k])] ;GB=self.X2G[tuple(self.BeamSeg[i].NodeX[k+1])]
                          cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
                          self.ins.cbeam(cbeam)
                          EID=EID+1
                      elif  i==0 and k==self.BeamSeg[i].nx-1:
                          continue

                      elif  i!=0 and k==0:
                          GA=self.X2G[tuple(self.BeamSeg[self.inp.BeamConn[i]].NodeX[-1])];GB=self.X2G[tuple(self.BeamSeg[i].NodeX[k])]
                          cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
                          self.ins.cbeam(cbeam)
                          EID=EID+1
                      else:
                          GA=self.X2G[tuple(self.BeamSeg[i].NodeX[k-1])] ;GB=self.X2G[tuple(self.BeamSeg[i].NodeX[k])]
                          cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
                          self.ins.cbeam(cbeam)
                          EID=EID+1

                self.idx=self.idx+self.BeamSeg[i].nx

        beams_materials(self)
        beams_properties(self)
        beams_grids(self)
        beams_elements(self)

    def masses(self):
        # Masses
        #========================================================

        if self.inp.conm1:
            idm=0
            for i in range(self.inp.NumBeams):
              for k in range(self.BeamSeg[i].nx):

                    Eid= self.idx+idm+(k)
                    RefGid=(k+1)+idm

                    CONM1=['CONM1',Eid,RefGid,0,self.inp.m11[i][k],self.inp.m21[i][k],self.inp.m22[i][k],
                           self.inp.m31[i][k],self.inp.m32[i][k],self.inp.m33[i][k], self.inp.m41[i][k],
                           self.inp.m42[i][k],self.inp.m43[i][k],self.inp.m44[i][k],self.inp.m51[i][k],
                           self.inp.m52[i][k],self.inp.m53[i][k],self.inp.m54[i][k],self.inp.m55[i][k],
                           self.inp.m61[i][k],self.inp.m62[i][k],self.inp.m63[i][k],self.inp.m64[i][k],
                           self.inp.m65[i][k],self.inp.m66[i][k]]
                    self.ins.conm1(CONM1)

              idm=idm+self.BeamSeg[i].nx

        if self.inp.conm2:

            idm=0
            for i in range(self.inp.NumBeams):
              for k in range(self.BeamSeg[i].nx):

                    Eid= self.idx+idm+(k)
                    RefGid=(k+1)+idm

                    CONM2=['CONM2',Eid,RefGid,0,self.inp.mass[i][k],
                           self.inp.X1[i][k],self.inp.X2[i][k],self.inp.X3[i][k],None,
                           self.inp.I11[i][k],self.inp.I21[i][k], self.inp.I22[i][k],
                           self.inp.I31[i][k],self.inp.I32[i][k],self.inp.I33[i][k]]
                    self.ins.conm2(CONM2)

              idm=idm+self.BeamSeg[i].nx

    def constraints(self):
        # Fix constraints
        #=====================================================

        #mesh.add_card_fields(['dd',1,3],'dd')
        #mesh.add_spcadd('[SPCADD,2,1]',SPCADD)
        if self.inp.Clamped:
          spc_all=2
          self.ins.spcadd(['SPCADD',spc_all]+self.inp.spc)
          C = self.inp.spc_dimen
          clamped_node=1
          for i in self.inp.spc:
              spc1=['SPC1',i,C,clamped_node]
              self.ins.spc(spc1)

    def forces(self):

      if self.inp.FORCE:
        for i in range(self.inp.numLoads):
          sid=2*(i+1)+1
          load1=['LOAD',sid,1.]
          for j in range(self.inp.numForce):

           sid=2*(i+1)+j*10
           G= self.X2G[tuple(self.BeamSeg[self.inp.gridF[i][j][0]].NodeX[self.inp.gridF[i][j][1]])]
           F= self.inp.Fl[i][j]
           force1=['FORCE',sid,G,0,F,self.inp.Fd[i][j][0],self.inp.Fd[i][j][1],self.inp.Fd[i][j][2]]
           self.ins.force(force1)
           load1.append(1.)
           load1.append(2*(i+1)+j*10)
          self.ins.load(load1,'LOAD')

      if self.inp.FORCE1:
        for i in range(self.inp.numLoads):
          lid=2*(i+1)+1
          load1=['LOAD',lid,1.]
          for j in range(self.inp.numForce):

           if self.inp.Fl is not None:
             gridpoint = self.BeamSeg[self.inp.gridF[i][j][0]].NodeX[self.inp.gridF[i][j][1]]+self.inp.Fd[i][j]
             gridId = (j+1)*1000+i
             node=['GRID',gridId,None,gridpoint[0],gridpoint[1],gridpoint[2],None,None,None]
             self.ins.grid(node)

             sid=2*(i+1)+(j+1)*100
             G= self.X2G[tuple(self.BeamSeg[self.inp.gridF[i][j][0]].NodeX[self.inp.gridF[i][j][1]])]
             F= self.inp.Fl[i][j]
             G1=G
             G2=gridId

             rbe2=['RBE2',(j+1)*100+i,G,123456,G2]
             self.ins.rbe2(rbe2)

             force1=['FORCE1',sid,G,F,G1,G2]
             self.ins.force1(force1)
             load1.append(1.)
             load1.append(sid)

           if self.inp.Ml is not None:
             gridpointm = self.BeamSeg[self.inp.gridF[i][j][0]].NodeX[self.inp.gridF[i][j][1]]+self.inp.Md[i][j]
             if np.allclose(gridpoint,gridpointm) is not True:
               gridId = (j+1)*1110+i
               node=['GRID',gridId,None,gridpointm[0],gridpointm[1],gridpointm[2],None,None,None]
               self.ins.grid(node)
               G= self.X2G[tuple(self.BeamSeg[self.inp.gridF[i][j][0]].NodeX[self.inp.gridF[i][j][1]])]
               G1=G
               G2=gridId

               rbe2=['RBE2',(j+1)*100+i,G,123456,G2]
               self.ins.rbe2(rbe2)

             sid=2*(i+1)+(j+1)*100+1
             M = self.inp.Ml[i][j]
             moment1=['MOMENT1',sid,G,M,G1,G2]
             self.ins.moment1(moment1)
             load1.append(1.)
             load1.append(sid)
          self.ins.load(load1)
          
    def run(self,masses=1,constraints=1,initials=0,forces=0):
         self.beams()
         if initials:
             self.beams_initials()
         if masses:
             self.masses()
         if constraints:
             self.constraints()
         if forces:
             self.forces()


# class try1:
#     def f(self,x,y):
#      def f1(self,x,y):
#         self.fx=x
#         print 'f1'
#      def f2(self):
#         #self.f1(4,5)
#         print self.fx*7
#      f1(self,x,y)
#      f2(self)

      # if inp.PRESSURE:

      #   for i in range(inp.numLoads):
      #     sid=2*(i+1)+1
      #     load1=['LOAD',sid,1.]
      #     for j in range(inp.N[0]-1):

      #      sid2=2000*(i+1)+j*10
      #      load1.append(1.)
      #      load1.append(sid2)
      #      pl1=['PLOAD1',sid2,j+1,'FZE','FR',0.,inp.Fp[i],1.,inp.Fp[i]]
      #      mesh.add_card(pl1,'PLOAD1')


      #     mesh.add_card(load1,'LOAD')



class WingBox_model:
    def __init__(self,inp,ins):
        self.inp = importlib.import_module(inp)
        self.ins = ins
        if self.inp.rotation:
            import intrinsic.Tools.transformations
            self.Rba=intrinsic.Tools.transformations.rotation_matrix(self.inp.rot_angle,self.inp.rot_direc)[:3, :3]

    # Tapper functions
    #===============================================================
    def f(self,x):

        Ly=self.inp.Lyr-(self.inp.Lyr-self.inp.Lyt)/self.inp.Lx*x
        return Ly


    def g(self,x):

        Lz=self.inp.Lzr-(self.inp.Lzr-self.inp.Lzt)/self.inp.Lx*x
        return Lz

    def box_initials(self,ND=50):
        # Eigenvector Card
        #==========================
        if self.inp.pch==0:
          SID = 1#method
          #ND = NumNodes*6
          eigrl = ['EIGRL', SID, None, None, ND, None, None, None,'MASS']
          self.ins.eigrl(eigrl)
    def box(self):

        #=======================================================================
        # Material cards
        #=======================================================================
        if self.inp.lumped:
            id_mat=1; mat1 = ['MAT1',id_mat,self.inp.Em,None,self.inp.Nu,None]
            self.ins.mat1(mat1)
        else:
            id_mat=1; mat1 = ['MAT1',id_mat,self.inp.Em,None,self.inp.Nu,self.inp.rho]
            self.ins.mat1(mat1)

        #============================================================================
        # Shell properties flat
        #============================================================================
        id_p=1; pshell1 = ['PSHELL',id_p,id_mat,self.inp.thickness,id_mat,None,id_mat]
        self.ins.pshell(pshell1)


        #=========================================================================
        # Nodes
        #=========================================================================
        for i in range(self.inp.nx):
            for j in range(self.inp.ny-1):

                x=i*self.inp.dlx
                y=self.f(x)/2 - self.f(x)/self.inp.Lyr*self.inp.dly*j + self.inp.tipy/self.inp.Lx*x
                z=self.g(x)/2+self.inp.tipz/self.inp.Lx*x
                if self.inp.rotation:
                    x,y,z=self.Rba.dot([x,y,z])
                Id=100000+1000*j+i
                node=['GRID',Id,None,x,y,z,None,None,None]
                self.ins.grids(node)

                x=i*self.inp.dlx
                y=-self.f(x)/2 + self.f(x)/self.inp.Lyr*self.inp.dly*j + self.inp.tipy/self.inp.Lx*x
                z=-self.g(x)/2+self.inp.tipz/self.inp.Lx*x
                if self.inp.rotation:
                    x,y,z=self.Rba.dot([x,y,z])
                Id=300000+1000*j+i
                node=['GRID',Id,None,x,y,z,None,None,None]
                self.ins.grids(node)

        for i in range(self.inp.nx):
            for k in range(self.inp.nz-1):

                x=i*self.inp.dlx
                y=-self.f(x)/2+self.inp.tipy/self.inp.Lx*x
                z= self.g(x)/2-self.g(x)/self.inp.Lzr*self.inp.dlz*k+ self.inp.tipz/self.inp.Lx*x
                if self.inp.rotation:
                    x,y,z=self.Rba.dot([x,y,z])
                Id=200000+1000*k+i
                node=['GRID',Id,None,x,y,z,None,None,None]
                self.ins.grids(node)

                x=i*self.inp.dlx
                y=self.f(x)/2+self.inp.tipy/self.inp.Lx*x
                z= -self.g(x)/2+self.g(x)/self.inp.Lzr*self.inp.dlz*k+ self.inp.tipz/self.inp.Lx*x
                if self.inp.rotation:
                    x,y,z=self.Rba.dot([x,y,z])
                Id=400000+1000*k+i
                node=['GRID',Id,None,x,y,z,None,None,None]
                self.ins.grids(node)

        #========================================================================
        #Elements
        #========================================================================
        h=1
        for i in range(self.inp.nx-1):
            for j in range(self.inp.ny-1):

                if j==self.inp.ny-2:
                    Id=h
                    G1=100000+1000*j+i
                    G2=200000+1000*(0)+i
                    G3=200000+1000*(0)+(i+1)
                    G4=100000+1000*j+(i+1)
                    quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
                    self.ins.cquad4(quad)
                    h=h+1
                else:
                    Id=h
                    G1=100000+1000*j+i
                    G2=100000+1000*(j+1)+i
                    G3=100000+1000*(j+1)+(i+1)
                    G4=100000+1000*j+(i+1)
                    quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
                    self.ins.cquad4(quad)
                    h=h+1


                if j==self.inp.ny-2:
                    Id=h
                    G1=300000+1000*(self.inp.ny-2)+i
                    G2=400000+1000*(0)+i
                    G3=400000+1000*(0)+(i+1)
                    G4=300000+1000*(self.inp.ny-2)+(i+1)
                    quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
                    self.ins.cquad4(quad)
                    h=h+1
                else:
                    Id=h
                    G1=300000+1000*j+i
                    G2=300000+1000*(j+1)+i
                    G3=300000+1000*(j+1)+(i+1)
                    G4=300000+1000*j+(i+1)
                    quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
                    self.ins.cquad4(quad)
                    h=h+1


        for i in range(self.inp.nx-1):
            for k in range(self.inp.nz-1):

                if k==self.inp.nz-2:
                    Id=h
                    G1=200000+1000*k+i
                    G2=300000+1000*(0)+i
                    G3=300000+1000*(0)+(i+1)
                    G4=200000+1000*k+(i+1)
                    quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
                    self.ins.cquad4(quad)
                    h=h+1
                else:
                    Id=h
                    G1=200000+1000*k+i
                    G2=200000+1000*(k+1)+i
                    G3=200000+1000*(k+1)+(i+1)
                    G4=200000+1000*k+(i+1)
                    quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
                    self.ins.cquad4(quad)
                    h=h+1
                if k==self.inp.nz-2:
                    Id=h
                    G1=400000+1000*(self.inp.nz-2)+i
                    G2=100000+1000*(0)+i
                    G3=100000+1000*(0)+(i+1)
                    G4=400000+1000*(self.inp.nz-2)+(i+1)
                    quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
                    self.ins.cquad4(quad)
                    h=h+1
                else:
                    Id=h
                    G1=400000+1000*k+i
                    G2=400000+1000*(k+1)+i
                    G3=400000+1000*(k+1)+(i+1)
                    G4=400000+1000*k+(i+1)
                    quad=['CQUAD4',Id,id_p,G1,G2,G3,G4,None,None]
                    self.ins.cquad4(quad)
                    h=h+1

    def asets(self,aset1):
        AsetId=[]
        for i in range(self.inp.na):
            x = self.inp.aset[i]
            y = self.inp.tipy/self.inp.Lx*x
            z = self.inp.tipz/self.inp.Lx*x
            if self.inp.rotation:
                x,y,z=self.Rba.dot([x,y,z])
            Id= 500000 + int(round(self.inp.aset[i]/self.inp.dlx))
            node=['GRID',Id,None,x,y,z,None,None,None]
            AsetId.append(Id)
            self.ins.grids(node)
            if aset1:
                self.ins.aset1(['ASET1',123456,Id])

    def mass_asets(self):
        for i in range(self.inp.na):
                Eid= 600000 + int(round(self.inp.aset[i]/self.inp.dlx))
                RefGid= 500000 + int(round(self.inp.aset[i]/self.inp.dlx))
                mass=self.inp.Mass[i]
                I11=self.inp.I[0,i];I21=self.inp.I[1,i];I22=self.inp.I[2,i];I31=self.inp.I[3,i];I32=self.inp.I[4,i];I33=self.inp.I[5,i]
                X1=0.0;X2=0.0;X3=0.0
                CONM2=['CONM2',Eid,RefGid,0,mass,X1,X2,X3,None,I11,I21,I22,I31,I32,I33]
                self.ins.conm2(CONM2)

    def interpolation_elements(self):
        #=======================================================================
        # RBE3s
        #=======================================================================
        for i in range(self.inp.na):

            Eid = 700000 + int(round(self.inp.aset[i]/self.inp.dlx))
            RefGid = 500000 + int(round(self.inp.aset[i]/self.inp.dlx))
            W1=1.
            RefC=123456
            C1=123
            RBE3 = ['RBE3',Eid,None,RefGid,RefC,W1,C1]

            for j in range(self.inp.ny-1):
             Gi1 = 100000+1000*j+int(round(self.inp.aset[i]/self.inp.dlx))
             RBE3.append(Gi1)
             Gi2 = 300000+1000*j+int(round(self.inp.aset[i]/self.inp.dlx))
             RBE3.append(Gi2)

            for k in range(self.inp.nz-1):
             Gi3 = 200000+1000*k+int(round(self.inp.aset[i]/self.inp.dlx))
             RBE3.append(Gi3)
             Gi4 = 400000+1000*k+int(round(self.inp.aset[i]/self.inp.dlx))
             RBE3.append(Gi4)

            Gm1=100000+1000*0+int(round(self.inp.aset[i]/self.inp.dlx))
            Cm1=123
            Gm2=300000+1000*0+int(round(self.inp.aset[i]/self.inp.dlx))
            Cm2=12
            Gm3=200000+1000*0+int(round(self.inp.aset[i]/self.inp.dlx))
            Cm3=1
            RBE3.append('UM')
            RBE3.append(Gm1)
            RBE3.append(Cm1)
            RBE3.append(Gm2)
            RBE3.append(Cm2)
            RBE3.append(Gm3)
            RBE3.append(Cm3)
            self.ins.rbe3(RBE3)

    def constraints(self):
        Sid=1
        spc = 2
        #mesh.add_spcadd(spc,[Sid])
        self.ins.spcadd(['SPCADD',spc,Sid])

        C=123456
        spc1=['SPC1',Sid,C]
        for j in range(self.inp.ny-1):
         Gi1 = 100000+1000*j
         spc1.append(Gi1)
         Gi2= 300000+1000*j
         spc1.append(Gi2)

        for k in range(self.inp.nz-1):
         Gi3 = 200000+1000*k
         spc1.append(Gi3)
         Gi4 = 400000+1000*k
         spc1.append(Gi4)

        self.ins.spc(spc1)

    def initial_conditions(self):
        # Initial conditions
        #===========================

        if self.inp.sol=='400' and self.inp.dynamic and self.inp.Velocity0:

            for vi in range(len(self.inp.vg)):
              Id = AsetId[vi]
              GId = self.ins.mesh.Node(Id).get_position()
              #print GId
              for d in range(3):

                  tic=['TIC',1,Id,d+1,0.,self.inp.fv(GId)[d]]
                  self.ins.tic(tic)


        if self.inp.sol=='400' and self.inp.dynamic  and self.inp.Displacement0:

            for ui in range(len(self.inp.ug)):
              Id =  AsetId[ui]
              GId = self.ins.mesh.Node(Id).get_position()
              for d in range(3):

                  tic=['TIC',1,Id,d+1,self.inp.fu(GId)[d]]
                  self.ins.tic(tic)

        if self.inp.sol=='400' and self.inp.dynamic  and self.inp.Acceleration0:

            for ai in range(len(self.inp.ag)):
              Id = self.ins.mesh.asets[0].node_ids[ai]
              GId = self.ins.mesh.Node(Id).get_position()
              for d in range(3):

                  tic=['TIC',1,Id,d+1,self.inp.fa(GId)[d]]
                  self.ins.tic(tic)

    def forces(self):

        for i in range(self.inp.numLoads):
          lid=2*(i+1)+1
          load1=['LOAD',lid,1.]
          for j in range(self.inp.numForce):

           if self.inp.Fl is not None and self.inp.FORCE1:
             gridpoint = self.ins.mesh.nodes[self.inp.gridF[i][j]].get_position()+self.inp.Fd[i][j]
             gridId = (j+1)*1000+i
             node=['GRID',gridId,None,gridpoint[0],gridpoint[1],gridpoint[2],None,None,None]
             self.ins.grids(node)

             sid=2*(i+1)+(j+1)*100
             G=self.inp.gridF[i][j]
             F= self.inp.Fl[i][j]
             G1=G
             G2=gridId

             rbe2=['RBE2',(j+1)*100+i,G,123456,G2]
             self.ins.rbe2(rbe2)

             force1=['FORCE1',sid,G,F,G1,G2]
             self.ins.force1(force1)
             load1.append(1.)
             load1.append(sid)

           if self.inp.Fl is not None and self.inp.FORCE:

             sid=2*(i+1)+(j+1)*100
             G=self.inp.gridF[i][j]
             F= self.inp.Fl[i][j]
             
             force1=['FORCE',sid,G,None,F,self.inp.Fd[i][j][0],self.inp.Fd[i][j][1],self.inp.Fd[i][j][2]]
             self.ins.force(force1)
             load1.append(1.)
             load1.append(sid)

           if self.inp.Ml is not None:
             gridpointm = self.ins.mesh.nodes[self.inp.gridF[i][j]].get_position()+self.inp.Md[i][j]
             if np.allclose(gridpoint,gridpointm) is not True:
               gridId = (j+1)*1110+i
               node=['GRID',gridId,None,gridpointm[0],gridpointm[1],gridpointm[2],None,None,None]
               self.ins.grids(node)
               G=self.inp.gridF[i][j]
               G1=G
               G2=gridId

               rbe2=['RBE2',(j+1)*100+i,G,123456,G2]
               self.ins.rbe2(rbe2)

             sid=2*(i+1)+(j+1)*100+1
             #print sid
             M = self.inp.Ml[i][j]
             moment1=['MOMENT1',sid,G,M,G1,G2]
             self.ins.moment1(moment1)
             load1.append(1.)
             load1.append(sid)
          self.ins.load(load1)
        if self.inp.dynamic:
            for i in range(self.inp.numLoads):
                lid=2*(i+1)+1
                dload=['DLOAD',7*(i+1)+1,1.,1.,i+1]
                tload1=['TLOAD1',i+1,lid,None,'LOAD',10]
                tabled1=['TABLED1',10]+[None for ix in range(7)]
                for ti in range(len(self.inp.tableti[i])):
                    tabled1.append(self.inp.tableti[i][ti][0])
                    tabled1.append(self.inp.tableti[i][ti][1])
                tabled1.append('ENDT')
                self.ins.dload(dload)
                self.ins.tload1(tload1)
                self.ins.tabled1(tabled1)
            if self.inp.sol=='400':    
                nlstep=['NLSTEP',1,self.inp.ti_max]+[None for i in range(6)]+['GENERAL']+[None for i in range(7)]+['FIXED',self.inp.ti_n]
                self.ins.nlstep(nlstep)
            if self.inp.sol=='109':    
                tstep=['TSTEP',1,self.inp.ti_n,self.inp.ti_max/self.inp.ti_n]
                self.ins.tstep(tstep)

          #LOAD
          #DLOAD,ID,1.,1.,TLOAD1
          #TLOAD1,ID,LOAD,,'LOAD',TABLED1
          #TABLED1

    def run(self,box_initials=1,box=1,asets=1,aset1=1,mass_asets=1,interpolation_elements=1,constraints=1,initial_conditions=0,forces=0):
         if box_initials:
             self.box_initials()
         if box:
             self.box()
         if asets:
             self.asets(aset1)
         if mass_asets:
             self.mass_asets()
         if interpolation_elements:
             self.interpolation_elements()
         if constraints:
             self.constraints()
         if initial_conditions:
             self.initial_conditions()
         if forces:
             self.forces()


class Aerodynamic_model:
    def __init__(self,inp,ins):
        if inp:
            self.inp = importlib.import_module(inp)
        self.ins = ins

    def surfaces(self):
        self.ins.aero(self.inp.aero)
        for si in range(self.inp.NumSurfaces):
            self.ins.caero1(self.inp.caero1[si])
            self.ins.paero1(self.inp.paero1[si])
            self.ins.spline(self.inp.spline67[si])
            self.ins.aelist(self.inp.aelist[si])
            self.ins.set1(self.inp.set1[si])

    @staticmethod
    def mkaero(model,machs,reduced_freqs,write_file):
        model.add_mkaero1(machs,reduced_freqs)
        path = '/'.join(write_file.split('/')[:-1])
        if not os.path.exists(path):
          os.makedirs(path)
        model.write_bdf(write_file)

    def executiveCD103(self,aa=1,gg=0):

        ecd = ""

        if aa:
            ecd += "assign output4='Phia.op4',formatted,UNIT=11\n"
            ecd += "assign output4='Ma.op4',formatted,UNIT=12\n"
            ecd += "assign output4='Ka.op4',formatted,UNIT=13\n"
        if gg:
            ecd += "assign output4='Phif.op4',formatted,UNIT=11\n"
            ecd += "assign output4='Mf.op4',formatted,UNIT=12\n"
            ecd += "assign output4='Kf.op4',formatted,UNIT=13\n"

        #NASTRAN NLINES=999999
        #NASTRAN BARMASS=1 (Allows the user to select the bar torsional mass moment
        #of inertia. If set to 0, request the pre-MSC Nastran 2004 (Default = 0).
        #If set to greater than 0, the torsional mass moment of inertia term is included
        #in the mass matrix formulation of bar elements. For both values of COUPMASS,
        #the torsional inertia is added.For COUPMASS = 1,
        #the axial mass will be consistent rather than coupled.)
        ecd += "SOL 103\n"
        ecd += "DIAG  20\n"
        ecd += "COMPILE SEMODES\n"
        ecd += "ALTER  'CALL.*SUPER3.*CASECC.*CASEM.*PHA1.*LAMA2'\n"
        if gg:
            ecd += "OUTPUT4 PHG,,,,//0/11///20\n"
            ecd += "OUTPUT4 MGG,,,,//0/12///20\n"
            ecd += "OUTPUT4 KGG,,,,//0/13///20\n"
        if aa:
            ecd += "OUTPUT4 PHA,,,,//0/11///20\n"
            ecd += "OUTPUT4 MAA,,,,//0/12///20\n"
            ecd += "OUTPUT4 KAA,,,,//0/13///20\n"
            ecd += "CEND"

        self.ins.executive_control_deck([ecd])

    def executiveCD_AICs(self,NumModes,phi0='Phi0.op4',phi1='Phi1.op4'):
        #ecd = "$EXECUTICE CASE CONTROL DECK"
        ecd = ""
        ecd += "assign OUTPUT4='Qs/Qhh%s.op4',formatted,UNIT=11\n"%NumModes
        ecd += "assign OUTPUT4='Qs/Qhj%s.op4',formatted,UNIT=12\n"%NumModes
        ecd += "assign INPUTT4='%s',formatted,UNIT=90"%phi0
        ecd += "\n"
        ecd += "assign INPUTT4= '%s',formatted,UNIT=91"%phi1
        ecd += "\n"
        ecd += "$NASTRAN NLINES=999999\n"
        ecd += "NASTRAN QUARTICDLM=1\n"
        # A value of 1 in QUARTICCDLM selects the new quartic formulation of the doublet lattic
        # kernel (N5KQ), while 0 selects the original quadratic form (Default = 0).
        ecd +="""SOL 146
$TIME 10000
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$       * * *   DMAP code by ANDREA CASTRICHINI  * * *
$                andrea.castrichini@gmail.com
$   K : AERODYNAMIC DOF = 2 * N AERO BOX
$   A : STRUCTURAL DOF  = 6 * N GRID
$   J : N AERO BOX
$   H : N MODES
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$ Retrieve the Spline Matrices form AERO0 and store them in DBALL
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COMPILE AERO0
ALTER 'CALL.*PLINOUT.*AECASE.*AEBGPDTS.*AEUSETS.*GPGK0.*GDGK0'
EQUIVX GPGK0/SPL_F_AK/ALWAYS
EQUIVX GDGK0/SPL_D_AK/ALWAYS $
CALL DBSTORE SPL_F_AK,,,,//111/112/'DBALL'/0 $
CALL DBSTORE SPL_D_AK,,,,//113/114/'DBALL'/0 $
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$ Retrieve the Matrices AJJ D12JK SKJ form PFAERO and store them in DBALL
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COMPILE PFAERO
ALTER 'AMG.*MKLIST,ACPT/'
TYPE PARM,,I,N,EXIST $
TYPE PARM,,CS,N,CK $
CALL DBFETCH /SPL_F_AK,,,,/111/112/0/0/S,EXIST $
CALL DBFETCH /SPL_D_AK,,,,/113/114/0/0/S,EXIST $
EQUIVX AJJT/A_JJT/ALWAYS $
EQUIVX SKJ/S_KJ/ALWAYS $
EQUIVX D1JK/D1_KJ/ALWAYS $
EQUIVX D2JK/D2_KJ/ALWAYS $
IF ( YESWKK ) THEN $
    EQUIVX WKK/W_KK/ALWAYS $
    MPYAD W_KK,S_KJ,/WS_KJ $
    EQUIVX WS_KJ/S_KJ/ALWAYS $
ENDIF $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$    EVALUATE THE QAA_ MATRIX:
$
$    Q_HH=Phi_HA*SPL_F_AK*S_KJ*INV(A_JJ)*D_JK*SPL_D_KA*Phi_AH
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
INPUTT4 /Phi_AH,,,,/1/91 $
INPUTT4 /Phi_AH1,,,,/1/91 $
$TRNSP Phi_AH/Phi_HA $
TRNSP Phi_AH1/Phi_HA $
TRNSP A_JJT/A_JJ $
TRNSP SPL_D_AK/SPL_D_KA $
DECOMP A_JJ/L_AJJ,U_AJJ,, $
CK = CMPLX(0.,KBAR) $
ADD5 D1_KJ,D2_KJ,,,/D_KJ//CK $
TRNSP D_KJ/D_JK $
MPYAD D_JK,SPL_D_KA,/D_JA $
FBS L_AJJ,U_AJJ,D_JA/Q_JA $
MPYAD S_KJ,Q_JA,/Q_KA $
MPYAD SPL_F_AK,Q_KA,/Q_AA $
MPYAD Phi_HA,Q_AA,/Q_HA $
MPYAD Q_HA,Phi_AH,/Q_HH $
OUTPUT4 Q_HH,,,,//0/11///9 $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$    EVALUATE THE QHJ_ MATRIX:
$
$    Q_HJ=Phi_HA*SPL_F_AK*S_KJ*INV(A_JJ)*D_JK*SPL_D_KA
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
PARAML A_JJ//'TRAILER'/1/S,N,COL $
MATGEN ,/ID/1/COL $
FBS L_AJJ,U_AJJ,ID/INVAJJ $
MPYAD S_KJ,INVAJJ,/Q_KJ $
MPYAD SPL_F_AK,Q_KJ,/Q_AJ $
MPYAD Phi_HA,Q_AJ,/Q_HJ $
OUTPUT4 Q_HJ,,,,//0/12///9 $
CEND"""
        self.ins.executive_control_deck([ecd])

    def caseCD_AICs(self,add_ccd=1):

        ccd  = ''
        ccd += 'METHOD = 910\n'
        ccd +='$SDAMP  = 920\n'
        ccd +='FREQ = 930\n'
        ccd +='$TSTEP = 940\n'
        ccd +='MONITOR = ALL\n'
        ccd +='RESVEC = NO\n'
        ccd +='K2GG=KAAX\n'
        ccd +='M2GG=MAAX\n'
        ccd +='SUBCASE 1\n'

        self.ins.case_control_deck(ccd,add_ccd)

    def bulkCD_AICs(self):

        self.ins.mesh.add_param('POST',0)
        self.ins.mesh.add_param('AUTOMSET','YES')
        #self.ins.add_param('BAILOUT',-1)
        # Bailout use to continue the run near singularities
        self.ins.mesh.add_param('WTMASS',1.0)
        self.ins.mesh.add_param('Q',1.)
        #self.ins.add_param('LMODES',4)
        #self.ins.add_card(['TABDMP1',920,'CRIT'],'TABDMP1')
        #Defines modal damping as a tabular function of natural frequency.
        self.ins.mesh.add_eigrl(910, v1=None, v2=None, nd=50, msglvl=0,
        maxset=None, shfscl=None, norm='MASS', options=None, values=None,
        comment='')
        self.ins.mesh.add_freq1(sid=930, f1=0.0,df=1.E-9, ndf=1, comment='')


class Aerodynamic_model2(Aerodynamic_model):

    def executiveCD_AICs(self,NumModes,phi0='Phi0.op4',phi1='Phi1.op4'):
        #ecd = "$EXECUTICE CASE CONTROL DECK"
        ecd = ""
        ecd += "assign OUTPUT4='Qs/Qah%s.op4',formatted,UNIT=11\n"%NumModes
        #ecd += "assign OUTPUT4='Qhj.op4',formatted,UNIT=12\n"
        #ecd += "assign INPUTT4='%s',formatted,UNIT=90"%phi0
        #ecd += "\n"
        ecd += "assign INPUTT4= '%s',formatted,UNIT=91"%phi1
        ecd += "\n"
        ecd += "$NASTRAN NLINES=999999\n"
        ecd += "NASTRAN QUARTICDLM=1\n"
        # A value of 1 in QUARTICCDLM selects the new quartic formulation of the doublet lattic
        # kernel (N5KQ), while 0 selects the original quadratic form (Default = 0).
        ecd +="""SOL 146
$TIME 10000
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$       * * *   DMAP code by ANDREA CASTRICHINI  * * *
$                andrea.castrichini@gmail.com
$   K : AERODYNAMIC DOF = 2 * N AERO BOX
$   A : STRUCTURAL DOF  = 6 * N GRID
$   J : N AERO BOX
$   H : N MODES
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$ Retrieve the Spline Matrices form AERO0 and store them in DBALL
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COMPILE AERO0
ALTER 'CALL.*PLINOUT.*AECASE.*AEBGPDTS.*AEUSETS.*GPGK0.*GDGK0'
EQUIVX GPGK0/SPL_F_AK/ALWAYS
EQUIVX GDGK0/SPL_D_AK/ALWAYS $
CALL DBSTORE SPL_F_AK,,,,//111/112/'DBALL'/0 $
CALL DBSTORE SPL_D_AK,,,,//113/114/'DBALL'/0 $
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$ Retrieve the Matrices AJJ D12JK SKJ form PFAERO and store them in DBALL
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
COMPILE PFAERO
ALTER 'AMG.*MKLIST,ACPT/'
TYPE PARM,,I,N,EXIST $
TYPE PARM,,CS,N,CK $
CALL DBFETCH /SPL_F_AK,,,,/111/112/0/0/S,EXIST $
CALL DBFETCH /SPL_D_AK,,,,/113/114/0/0/S,EXIST $
EQUIVX AJJT/A_JJT/ALWAYS $
EQUIVX SKJ/S_KJ/ALWAYS $
EQUIVX D1JK/D1_KJ/ALWAYS $
EQUIVX D2JK/D2_KJ/ALWAYS $
IF ( YESWKK ) THEN $
    EQUIVX WKK/W_KK/ALWAYS $
    MPYAD W_KK,S_KJ,/WS_KJ $
    EQUIVX WS_KJ/S_KJ/ALWAYS $
ENDIF $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$    EVALUATE THE QAA_ MATRIX:
$
$    Q_HH=Phi_HA*SPL_F_AK*S_KJ*INV(A_JJ)*D_JK*SPL_D_KA*Phi_AH
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
INPUTT4 /Phi_AH,,,,/1/91 $
INPUTT4 /Phi_AH1,,,,/1/91 $
$TRNSP Phi_AH/Phi_HA $
TRNSP Phi_AH1/Phi_HA $
TRNSP A_JJT/A_JJ $
TRNSP SPL_D_AK/SPL_D_KA $
DECOMP A_JJ/L_AJJ,U_AJJ,, $
CK = CMPLX(0.,KBAR) $
ADD5 D1_KJ,D2_KJ,,,/D_KJ//CK $
TRNSP D_KJ/D_JK $
MPYAD D_JK,SPL_D_KA,/D_JA $
FBS L_AJJ,U_AJJ,D_JA/Q_JA $
MPYAD S_KJ,Q_JA,/Q_KA $
MPYAD SPL_F_AK,Q_KA,/Q_AA $
$MPYAD Phi_HA,Q_AA,/Q_HA $
MPYAD Q_AA,Phi_AH/Q_AH $
OUTPUT4 Q_AH,,,,//0/11///9 $
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$    EVALUATE THE QHJ_ MATRIX:
$
$    Q_HJ=Phi_HA*SPL_F_AK*S_KJ*INV(A_JJ)*D_JK*SPL_D_KA
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$PARAML A_JJ//'TRAILER'/1/S,N,COL $
$MATGEN ,/ID/1/COL $
$FBS L_AJJ,U_AJJ,ID/INVAJJ $
$MPYAD S_KJ,INVAJJ,/Q_KJ $
$MPYAD SPL_F_AK,Q_KJ,/Q_AJ $
$MPYAD Phi_HA,Q_AJ,/Q_HJ $
$OUTPUT4 Q_HJ,,,,//0/12///9 $
CEND"""

        self.ins.executive_control_deck([ecd])
    
# class Human:

#     def __init__(self):
#         self.name = 'Guido'

#     def f(self,x):
#         return x**2

#     def talk(self):
#         #self.name+='k'
#         print self.f(4)

# if __name__ == '__main__':
#     guido = Human()
#     print guido.name
#     print guido.head.talk()
