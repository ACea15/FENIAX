import os
import numpy as np
import unittest
import Functions
import pdb


for i in range(len(os.getcwd().split('/'))):
      if os.getcwd().split('/')[-i-1]=='FEM4INAS':
        feminas_dir="/".join(os.getcwd().split('/')[0:-i])


class TestCase_Tony2(unittest.TestCase):


    def test_gamma12(self):
        X1,gamma1lT=Functions.reading_gamma('./ModesData/gamma1.txt')
        gamma1=np.load(feminas_dir+'/Models'+'/Tony_Flying_Wing'+'/Test/Results_modes/gamma1_30.npy')
        err1,err1r = Functions.err_gamma1(X1,gamma1,gamma1lT)
        self.assertAlmostEqual(err1,0.,places=6)
        self.assertAlmostEqual(err1r,0.,places=3)

    def test_gamma22(self):
        X2,gamma2lT=Functions.reading_gamma('./ModesData/gamma2.txt')
        gamma2=np.load(feminas_dir+'/Models'+'/Tony_Flying_Wing'+'/Test/Results_modes/gamma2_30.npy')
        err2,err2r = Functions.err_gamma2(X2,gamma2,gamma2lT)
        self.assertAlmostEqual(err2,0.,places=6)
        self.assertAlmostEqual(err2r,0.,places=3)

if __name__ == '__main__':
    unittest.main()
