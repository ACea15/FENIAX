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
wm1.run(box_initials=0,box=1,asets=0,aset1=0,mass_asets=0,interpolation_elements=0,constraints=1,initial_conditions=0,forces=0)
m1.write_model(model='/m1.bdf')

mesh1a=BDF(debug=True,log=None)
m1a = mgen.Model('wing_inp',mesh1a)
wm1a = mgen.WingBox_model('wing_inp',m1a)
wm1a.run(box_initials=0,box=0,asets=1,aset1=1,mass_asets=0,interpolation_elements=1,constraints=0,initial_conditions=0,forces=0)
m1a.write_model(model='/m1a.bdf')
m1a.write_includes(['m1.bdf'],model='/m1a.bdf')


mesh2=BDF(debug=True,log=None)
m2 = mgen.Model('wing_inp',mesh2)
wm2 = mgen.WingBox_model('wing_inp',m2)
wm2.run(box_initials=1,box=0,asets=1,aset1=0,mass_asets=0,interpolation_elements=1,constraints=0,initial_conditions=0,forces=0)
m2.executive_control_deck()
m2.case_control_deck(setx=range(500001,500001+30))
m2.write_model(model='/m1_103.bdf')
m2.write_includes(['m1.bdf'],model='/m1_103.bdf')

mesh2d=BDF(debug=True,log=None)
m2d = mgen.Model('wing_inp',mesh2d)
wm2d = mgen.WingBox_model('wing_pch',m2d)
am2d = mgen.Aerodynamic_model('wing_inp',m2d)
wm2d.run(box_initials=1,box=0,asets=1,aset1=0,mass_asets=0,interpolation_elements=1,constraints=0,initial_conditions=0,forces=0)
am2d.executiveCD103()
#m2d.case_control_deck(ccd='METHOD = 1\n',add_ccd=1)
m2d.case_control_deck()
#m2d.eigr({'sid':1, 'method':'LAN', 'f1':None, 'f2':None, 'ne':None, 'nd':30,
#'norm':'MASS', 'G':None, 'C':None})
m2d.eigrl(['EIGRL',1,None,None,60,None,None,None,'MASS'])
model='/m1_103d.bdf'
m2d.write_model(model=model)
m2d.write_includes(['m1.bdf'],model=model)
#if run_nastran:
#    subprocess.call("msc20160 nastran %s scr=yes" %(m2.path+model),shell=True,executable='/bin/bash',cwd=m2.path)


mesh2p=BDF(debug=True,log=None)
m2p = mgen.Model('wing_pch',mesh2p)
wm2p = mgen.WingBox_model('wing_pch',m2p)
wm2p.run(box_initials=1,box=0,asets=1,aset1=0,mass_asets=0,interpolation_elements=1,constraints=0,initial_conditions=0,forces=0)
m2p.executive_control_deck()
m2p.case_control_deck()
m2p.write_model(model='/m1_103p.bdf')
m2p.write_includes(['m1.bdf'],model='/m1_103p.bdf')


mesh2a=BDF(debug=True,log=None)
m2a = mgen.Model('wing_inp',mesh2a)
wm2a = mgen.WingBox_model('wing_inp',m2a)
wm2a.run(box_initials=1,box=0,asets=0,aset1=0,mass_asets=0,interpolation_elements=0,constraints=0,initial_conditions=0,forces=0)
m2a.executive_control_deck()
m2a.case_control_deck(setx=range(500001,500001+30))
m2a.write_model(model='/m1a_103.bdf')
m2.write_includes(['m1a.bdf'],model='/m1a_103.bdf')

#EXTSEOUT(STIFFNESS ASMBULK EXTID = 10 MATOP4=18)



mesh3=BDF(debug=True,log=None)
m3 = mgen.Model('force_inp',mesh3)
wm3 = mgen.WingBox_model('force_inp',m3)
wm3.run(box_initials=0,box=0,asets=1,aset1=0,mass_asets=0,interpolation_elements=1,constraints=0,initial_conditions=0,forces=1)
m3.executive_control_deck()
m3.case_control_deck(setx=range(500001,500001+30))
m3.write_model(model='/m1_400mg.bdf')
m3.write_includes(['m1.bdf'],model='/m1_400mg.bdf')

# mesh4=BDF(debug=True,log=None)
# m4 = mgen.Model('pressure_inp',mesh4)
# wm4 = mgen.WingBox_model('pressure_inp',m4)
# wm4.run(box_initials=0,box=0,asets=1,aset1=0,mass_asets=0,interpolation_elements=0,constraints=0,initial_conditions=0,forces=1)
# m4.executive_control_deck()
# m4.case_control_deck(setx=range(500001,500001+25))
# m4.write_model(model='/m1_400p.bdf')
# m4.write_includes(['m1.bdf'],model='/m1_400p.bdf')
