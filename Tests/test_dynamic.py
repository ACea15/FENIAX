import os
import numpy as np
import unittest
import Functions
import pickle
import datetime
import subprocess
import Utils.op2reader as op2r


for i in range(len(os.getcwd().split('/'))):
      if os.getcwd().split('/')[-i-1]=='FEM4INAS':
        feminas_dir="/".join(os.getcwd().split('/')[0:-i])

        
def running(confi,sh_file,folder='results_'):

      now = datetime.datetime.now()
      date = "%s-%s-%s_%s-%s"%(now.day,now.month,now.year,now.hour,now.minute)
      save_folder = folder+date
      with open(confi,'r') as cfile:
            lines=cfile.readlines()
            ifyes=0
            for li in range(len(lines)):
                  if lines[li].split('=')[0]=='save_folder':
                        lines[li] = "save_folder='%s'"%save_folder
                        ifyes=1
            if not ifyes:
                  lines.append("save_folder='%s'"%save_folder)
            with open(confi,'w') as cfile2:
               cfile2.writelines(lines)
      cwd='/'.join(confi.split('/')[:-1])
      subprocess.call(sh_file,shell=True,cwd=cwd)
      return save_folder


# class TestCase_RafaBeam(unittest.TestCase):
#     def test_rafabeam(self):
#         #results=feminas_dir+'/Models'+'/RafaBeam30_lum'+'/Test/Results/F0'
#         results=feminas_dir+'/Models'+'/RafaBeam30_lum'+'/Resultstest/F0'
#         with open (results+'/Sol_%s'%109, 'rb') as fp:
#               #[ra0,ra,Rab,strain,kappa]  = pickle.load(fp)
#               [ra0,ra,Rab]  = pickle.load(fp)
#         rt = np.load('DynamicData/rafabeam.npy')

#         points = [1,10,20,-1]
#         err = Functions.err_rafabeam(ra,rt,points)
#         derr = [1e-3,0.014,0.007]
#         for i in range(3):
#               for j in range(len(points)):
#                     self.assertTrue(abs(err[i][j])<derr[i])

class TestCase_RafaBeam(unittest.TestCase):
    def test_rafabeam(self):
        results = running('./Models/RafaBeam/confi.py',"./test1.sh",folder='V30y/results_')
        #results = 'V30y/results_21-3-2019_11-43'
        results=feminas_dir+'/Models'+'/RafaBeam30_lum'+'/Test/'+results
        with open (results+'/Solv_%s'%109, 'rb') as fp:
              #[ra0,ra,Rab,strain,kappa]  = pickle.load(fp)
              [ra0,ra,Rab]  = pickle.load(fp)
        rt = np.load('DynamicData/rafabeam.npy')

        points = [1,10,20,-1]
        err = Functions.err_rafabeam(ra,rt,points)
        derr = [1e-3,0.014,0.007]
        for i in range(3):
              for j in range(len(points)):
                    self.assertTrue(abs(err[i][j])<derr[i])

class TestCase_FFB(unittest.TestCase):
      def test_cg2d(self):
          results = running('./Models/Hesse/confi2d.py',"./run2d.sh",folder='2D/results_')
          #results = "/media/acea/work/projects/FEM4INAS/Models/Hesse_25/Test/2D/results_17-5-2023_20-47/"
          results=feminas_dir+'/Models'+'/Hesse_25'+'/Test/'+results
          #results=feminas_dir+'/Models'+'/Hesse_25'+'/Test/2D/'+'results_6-10-2019_19-43'
          Cg=np.load(results+'/Cg_150.npy')
          ti=np.load(results+'/ti_150.npy')
          err=Functions.Hesse25_cgerr2d(Cg,ti,tl=None)
          print err
          self.assertTrue(err<5e-4)

      def test_cg3d(self):
          results = running('./Models/Hesse/confi3d.py',"./run3d.sh",folder='3D/results_')
          results=feminas_dir+'/Models'+'/Hesse_25'+'/Test/'+results
          Cg=np.load(results+'/Cg_150.npy')
          ti=np.load(results+'/ti_150.npy')
          err=Functions.Hesse25_cgerr3d(Cg,ti,tl=None)
          print err
          self.assertTrue(err<5e-3)

class TestCase_SailPlaneWing(unittest.TestCase):
      def test_guyan(self):
          results = running('./Models/WingSailPlane/confi_maing.py',"./rung.sh",folder='results_')
          results=feminas_dir+'/Models'+'/wingSP'+'/Test/'+results
          n1 = op2r.NastranReader('/media/acea/work/projects/FEM4INAS/Models/wingSP/nastran/calcnew/angle/wing400d',
                                  '/media/acea/work/projects/FEM4INAS/Models/wingSP/nastran/calcnew/angle/wing400dread')
          n1.readModel()
          tn,rn = n1.position()
          with open (results+'/Solv_%s'%53, 'rb') as fp:
              [ra0,ra,Rab] = pickle.load(fp)
          err=Functions.err_SailPlaneWing(ra,rn)
          print err
          self.assertTrue(err<1.02e-3)

      def test_kidder(self):
          results = running('./Models/WingSailPlane/confi_maini.py',"./runi.sh",folder='resultsi_')
          results=feminas_dir+'/Models'+'/wingSP'+'/Test/'+results
          n1 = op2r.NastranReader('/media/acea/work/projects/FEM4INAS/Models/wingSP/nastran/calcnew/angle/wing400d',
                                  '/media/acea/work/projects/FEM4INAS/Models/wingSP/nastran/calcnew/angle/wing400dread')
          n1.readModel()
          tn,rn = n1.position()
          with open (results+'/Solv_%s'%53, 'rb') as fp:
              [ra0,ra,Rab] = pickle.load(fp)
          err=Functions.err_SailPlaneWing(ra,rn)
          print err
          self.assertTrue(err<1.0009e-3)


class TestCase_DoublePendulum(unittest.TestCase):
      def test_free(self):
          results = running('./Models/DoublePendulum/confi_main.py',"./rung.sh",folder='results_double')
          results=feminas_dir+'/Models'+'/DPendulum'+'/Test/'+results
          with open (results+'/Solv_3' , 'rb') as fp:
                ra1,ra2  = pickle.load(fp)
          rth=np.load('./DynamicData/DPendulum_r_theo.npy')
          err=Functions.err_DoublePendulum(rth,ra2)
          print 'The normed-error in the double pendulum is %s' % err
          self.assertTrue(err<9.1e-4)

      def test_fixedHinge(self):
          results = running('./Models/DoublePendulum/confi_main.py',"./rungfix.sh",folder='results_fixed')
          results=feminas_dir+'/Models'+'/DPendulum'+'/Test/'+results
          with open (results+'/Solv_3' , 'rb') as fp:
                ra1,ra2  = pickle.load(fp)
          rth=np.load('./DynamicData/DPendulum_r_fixed.npy')
          err=Functions.err_DoublePendulumFixed(rth,ra2)
          print 'The normed-error in the pendulum is %s' % err
          self.assertTrue(err<1.8e-3)

          
#results=feminas_dir+'/Models'+'/Hesse_25'+'/results'
#with open (results+'/Solv_%s'%150, 'rb') as fp:
#      [ra0v,rav,Rabv]  = pickle.load(fp)
#Cg=np.load(results+'/Cg150.npy')
#results=feminas_dir+'/Models'+'/Hesse_25'+'/2D/results_14-4-2019_12-30/'
#with open (results+'/Solq_%s'%150, 'rb') as fp:
#      [ra0q,raq,Rabq,Q]  = pickle.load(fp)
#Cg=np.load(results+'Cg_50.npy')

if __name__ == '__main__':
    unittest.main()
