import os
import numpy as np
import unittest
import Functions
import pickle
import datetime
import subprocess
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
      subprocess.call('chmod +x '+sh_file,shell=True,cwd=cwd)
      subprocess.call('./'+sh_file,shell=True,cwd=cwd)
      return save_folder

# now = datetime.datetime.now()
# date = "%s-%s-%s_%s-%s"%(now.day,now.month,now.year,now.hour,now.minute)

# save_folder = 'results_'+date
# with open('./Models/Simo/Simo45_15/confi_simo_dead.py','r') as cfile:
#       lines=cfile.readlines()
#       ifyes=0
#       for li in range(len(lines)):
#             if lines[li].split('=')[0]=='save_folder':
#                   lines[li] = "save_folder='%s'"%save_folder
#                   ifyes=1
#       if not ifyes:
#             lines.append("save_folder='%s'"%save_folder)
#       with open('./Models/Simo/Simo45_15/confi_simo_dead.py','w') as cfile2:
#          cfile2.writelines(lines)

# subprocess.call("./test-dead.sh",shell=True,cwd='./Models/Simo/Simo45_15')

# class TestCase_Simo(unittest.TestCase)
# def test_SimoMoment(self):
# def test_SimoFollower(self):
# def test_SimoDead(self):

# class TestCase_Ebner(unittest.TestCase)
# def test_Ebner(self):

# class TestCase_Argyris(unittest.TestCase)
# def test_ArgyrisCantilever(self):
# def test_ArgyrisCantileverPress(self):
# def test_ArgyrisFrame(self):


class TestCase_Tony(unittest.TestCase):
    def __init__(self,*args,**kwargs):
          unittest.TestCase.__init__(self,*args, **kwargs)
          self.results = running('./Models/Tony_Flying_Wing/confi_wang.py',"test1.sh")
          self.results=feminas_dir+'/Models'+'/Tony_Flying_Wing'+'/Test/'+self.results
       
    def test_gamma12(self):
        X1,gamma1lT=Functions.reading_gamma('./ModesData/gamma1.txt')
        gamma1=np.load(self.results+'/Results_modes/gamma1_30.npy')
        err1,err1r = Functions.err_gamma1(X1,gamma1,gamma1lT)
        self.assertAlmostEqual(err1,0.,places=6)
        self.assertAlmostEqual(err1r,0.,places=3)

    def test_gamma22(self):
        X2,gamma2lT=Functions.reading_gamma('./ModesData/gamma2.txt')
        gamma2=np.load(self.results+'/Results_modes/gamma2_30.npy')
        err2,err2r = Functions.err_gamma2(X2,gamma2,gamma2lT)
        self.assertAlmostEqual(err2,0.,places=6)
        self.assertAlmostEqual(err2r,0.,places=3)


class TestCase_Argyris(unittest.TestCase):
    def test_ArgyrisCantilever(self):
        results = running('./Models/Argyris/ArgyrisBeam_25/confi_argyrisbeam.py',"test1.sh",folder='EndLoad/results_')
        results=feminas_dir+'/Models'+'/ArgyrisBeam_25'+'/Test/'+results
        with open (results+'/Sols_%s'%150 , 'rb') as fp:
              [ra0,ra,Rab,strain,kappa]  = pickle.load(fp)
        with open ('StaticData'+'/ArgyrisBeamSol_%s'%150 , 'rb') as fp:
              [ra0C,raC,RabC,strainC,kappaC]  = pickle.load(fp)

        for i in range(6):
              print "Argyris"
              print np.linalg.norm(raC[0][:,2*i+1]-ra[0][:,i])/np.linalg.norm(raC[0][:,2*i+1])
              self.assertTrue((np.linalg.norm(raC[0][:,2*i+1]-ra[0][:,i])/np.linalg.norm(raC[0][:,2*i+1]))<2.5e-7)
              print "#####"
    def test_ArgyrisFrame2d(self):
        results = running('./Models/Argyris/ArgyrisFrame_20/confi_argyrisframe_2d.py',"test1.sh",folder='2DProblem/results_')
        results=feminas_dir+'/Models'+'/ArgyrisFrame_20'+'/Test/'+results
        with open (results+'/Sols_%s'%120, 'rb') as fp:
              [ra0,ra,Rab,strain,kappa]  = pickle.load(fp)
        ut = np.load('StaticData/ArgyrisFrame2d_ut.npy')

        err1 = [3.7e-7,1.5e-6,1.2e-5,3e-5,8.2e-5,2.5e-4]
        err2 = [3.1e-7,1.2e-6,8.5e-6,2.3e-5,1.5e-4,4.6e-4]
        for i in range(6):
              self.assertTrue(np.linalg.norm(ut[i,:11]-ra[0][:,i])/np.linalg.norm(ut[i,:11])<err1[i])
              self.assertTrue(np.linalg.norm(ut[i,10:]-ra[1][:,i])/np.linalg.norm(ut[i,10:])<err2[i])

    def test_ArgyrisFrame3d(self):
        results = running('./Models/Argyris/ArgyrisFrame_20/confi_argyrisframe_3d.py',"test2.sh",folder='3DProblem/results_')
        results=feminas_dir+'/Models'+'/ArgyrisFrame_20'+'/Test/'+results
        with open (results+'/Sols_%s'%120, 'rb') as fp:
              [ra0,ra,Rab,strain,kappa]  = pickle.load(fp)
        ut = np.load('StaticData/ArgyrisFrame3d_ut.npy')

        err1 = [2e-5,6e-5,2e-4,3e-4,5e-4,7e-4]
        err2 = [2e-5,7e-5,2e-4,3e-4,6e-4,9e-4]
        for i in range(6):
              self.assertTrue(np.linalg.norm(ut[i,:11]-ra[0][:,2*i])/np.linalg.norm(ut[i,:11])<err1[i])
              self.assertTrue(np.linalg.norm(ut[i,10:]-ra[1][:,2*i])/np.linalg.norm(ut[i,10:])<err2[i])


class TestCase_Ebner(unittest.TestCase):
      def test_Ebner(self):
        results = running('./Models/Ebner/confi_ebner.py',"test.sh")
        results=feminas_dir+'/Models'+'/Ebner'+'/Test/'+results
        with open (results+'/Sols_%s'%108, 'rb') as fp:
              [ra0,ra,Rab,strain,kappa]  = pickle.load(fp)

        ut = np.load('./StaticData/Ebner_ut.npy')
        print "Ebner"
        print np.linalg.norm(ut[0][:10]-ra[0][:,-1])/np.linalg.norm(ut[0][:10])
        print np.linalg.norm(ut[0][9:]-ra[1][:,-1])/np.linalg.norm(ut[0][9:])
        print "######"
        self.assertTrue(np.linalg.norm(ut[0][:10]-ra[0][:,-1])/np.linalg.norm(ut[0][:10])<5.5e-4)
        self.assertTrue(np.linalg.norm(ut[0][9:]-ra[1][:,-1])/np.linalg.norm(ut[0][9:])<3.2e-4)


class TestCase_Simo(unittest.TestCase):

    # def test_Simo_Moment(self):
    #     results=feminas_dir+'/Models'+'/Simo_Moment'+'/Test/Results_F'
    #     with open (results+'/Sols_%s'%60, 'rb') as fp:
    #           [ra0,ra,Rab,strain,kappa]  = pickle.load(fp)

    #     ut = np.load('./StaticData/Ebner_ut.npy')

    #     self.assertTrue(np.linalg.norm(ut[0][:10]-ra[0][:,-1])/np.linalg.norm(ut[0][:10])<5.5e-4)
    #     self.assertTrue(np.linalg.norm(ut[0][9:]-ra[1][:,-1])/np.linalg.norm(ut[0][9:])<3e-4)

    def test_Simo45Follower(self):
        #results=feminas_dir+'/Models'+'/Simo45_15'+'/Test/Results_Ffollower'
        results = running('./Models/Simo/Simo45_15/confi_simo_foll.py',"test-foll.sh",folder='FollowerForce/results_')
        #results = "results_18-3-2019_17-55"
        results=feminas_dir+'/Models'+'/Simo45_15'+'/Test/'+results
        with open (results+'/Sols_%s'%90 , 'rb') as fp:
              [ra0,ra,Rab,strain,kappa]  = pickle.load(fp)
        with open ('StaticData'+'/Simo45Sol_Foll%s'%120, 'rb') as fp:
              [ra0C,raC,RabC,strainC,kappaC]  = pickle.load(fp)

        err = Functions.err_static_tip_simo45(ra,raC)
        errt = [1e-10, 3.9e-4,1.2e-3,1.5e-3,3.3e-3,7.3e-3,1.3e-2,1.9e-2, 2.5e-2, 3e-2, 3.5e-2]
        for i in range(len(err)):
              self.assertTrue(err[i]<errt[i])

    def test_Simo45Dead(self):
        results = running('./Models/Simo/Simo45_15/confi_simo_dead.py',"test-dead.sh",folder="DeadForce/results_")
        #results = 'results_18-3-2019_16-51'
        results=feminas_dir+'/Models'+'/Simo45_15'+'/Test/'+results
        #results=feminas_dir+'/Models'+'/Simo45_15'+'/Test/Results_Fdead'
        with open (results+'/Sols_%s'%90 , 'rb') as fp:
              [ra0,ra,Rab,strain,kappa]  = pickle.load(fp)
        with open ('StaticData'+'/Simo45Sol_Dead%s'%120, 'rb') as fp:
              [ra0C,raC,RabC,strainC,kappaC]  = pickle.load(fp)

        err = Functions.err_static_tip_simo45(ra,raC)
        errt = [1e-10, 1.5e-4,2.9e-4,3.8e-4,4.4e-4,4.9e-4,5.3e-4,5.5e-4,5.8e-4,6e-4,6.2e-2]
        for i in range(len(err)):
              self.assertTrue(err[i]<errt[i])


class TestCase_SailPlane(unittest.TestCase):

    def test_tipfollower(self):
        #results=feminas_dir+'/Models'+'/Simo45_15'+'/Test/Results_Ffollower'
        results = running('./Models/SailPlane/confi_static.py',"run_static.sh",folder='Static/results_')
        results=feminas_dir+'/Models'+'/SailPlane'+'/Test/'+results
        with open (results+'/Sols_%s'%50 , 'rb') as fp:
              [ra0,ra,Rab,strain,kappa]  = pickle.load(fp)
        rn=np.load('./StaticData/SailPlane_rt.npy')
        with open (results+'/Results_modes/Geometry', 'rb') as fp:
              BeamSeg,NumNode,NumNodes,DupNodes,inverseconn = pickle.load(fp)
        err1,err2 = Functions.err_SailPlane_static(ra,rn,BeamSeg)
        errt = [5e-5 for i in range(6)]
        print '######################################################################'
        for i in range(len(err1)):
              print 'The error in the static follower tip force of the sail plane is %s'%err1[i]
              self.assertTrue(err1[i]<errt[i])
        print '######################################################################'
if __name__ == '__main__':
    unittest.main()
