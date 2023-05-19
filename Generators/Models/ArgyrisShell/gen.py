import pdb
import subprocess
import numpy as np
import Generators.modelGen as mgen
reload(mgen)
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF

mesh1=BDF(debug=True,log=None)
m1 = mgen.Model('wing_inp',mesh1)
wm1 = mgen.WingBox_model('wing_inp',m1)
wm1.run(asets=0)
m1.write_model()

mesh2=BDF(debug=True,log=None)
m2 = mgen.Model('wing_inp',mesh2)
m2.executive_control_deck()
m2.case_control_deck()
m2.write_model(model='/m1_103.bdf')
m2.write_includes(['m1.bdf'],model='/m1_103.bdf')

mesh3=BDF(debug=True,log=None)
m3 = mgen.Model('force_inp',mesh3)
wm3 = mgen.WingBox_model('force_inp',m3)
wm3.run(box_initials=0,box=0,asets=1,aset1=0,mass_asets=0,interpolation_elements=0,constraints=0,initial_conditions=0,forces=1)
m3.executive_control_deck()
m3.case_control_deck(setx=range(500001,500001+25))
m3.write_model(model='/m1_400.bdf')
m3.write_includes(['m1.bdf'],model='/m1_400.bdf')

mesh4=BDF(debug=True,log=None)
m4 = mgen.Model('pressure_inp',mesh4)
wm4 = mgen.WingBox_model('pressure_inp',m4)
wm4.run(box_initials=0,box=0,asets=1,aset1=0,mass_asets=0,interpolation_elements=0,constraints=0,initial_conditions=0,forces=1)
m4.executive_control_deck()
m4.case_control_deck(setx=range(500001,500001+25))
m4.write_model(model='/m1_400p.bdf')
m4.write_includes(['m1.bdf'],model='/m1_400p.bdf')
