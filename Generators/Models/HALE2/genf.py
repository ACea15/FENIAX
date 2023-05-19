import pdb
import subprocess
import numpy as np
import Generators.modelGen as mgen
import Utils.FEM_MatrixBuilder as femmb
import Utils.common as common
import Utils.GridOrder as gridorder
from pyNastran.op2.op2 import OP2
from pyNastran.bdf.bdf import BDF
import importlib
import Runs.Torun
Runs.Torun.torun = 'HaleX1x'
Runs.Torun.variables = 'V'
run_V = 0
run_nastran = 1
run_model=1
run_fems=1
run_aerodynamics=1
run_AICs=1
if run_V:
    V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
else:
    class variables:
        pass
    V=variables()
    V.NumModes=5
################################################################################
# BULK MODEL
################################################################################
if run_model:
    mesh1=BDF(debug=True,log=None)
    m1 = mgen.Model('beam_inpf',mesh1)
    bm1 = mgen.Beam_model('beam_inpf',m1)
    bm1.run(masses=1,constraints=1,initials=0,forces=0)
    m1.write_model(model='/model.bdf')
    node_id2position={}
    nids=m1.mesh.node_ids[0:]
    for ni in nids:
        node_id2position[ni] = m1.mesh.nodes[ni].get_position()
################################################################################

################################################################################
# STIFFNESS AND MASS MATRICES
################################################################################
if run_fems:
    mesh2=BDF(debug=True,log=None)
    m2 = mgen.Model('beam_inpf',mesh2)
    am2 = mgen.Aerodynamic_model('aero_inp',m2)
    am2.executiveCD103()
    m2.case_control_deck(ccd='METHOD = 1\nDISP = ALL\n',add_ccd=1)
    #m2.eigr({'sid':1, 'method':'LAN', 'f1':None, 'f2':None, 'ne':None, 'nd':50,
    #'norm':'MASS', 'G':None, 'C':None})
    m2.eigrl(['EIGRL',1,None,None,60,None,None,None,'MASS'])
    model='/model103.bdf'
    m2.write_model(model=model)
    m2.write_includes(['model.bdf'],model=model)
    if run_nastran:
        subprocess.call("msc20160 nastran %s scr=yes" %(m2.path+model),shell=True,executable='/bin/bash',cwd=m2.path)
    import time
    time.sleep(10) 
################################################################################

################################################################################
# AERODYNAMICS
################################################################################
if run_aerodynamics:
    mesh3=BDF(debug=True,log=None)
    m3 = mgen.Model('beam_inpf',mesh3)
    am3 = mgen.Aerodynamic_model('aero_inp',m3)
    am3.surfaces()
    m3.write_model(model='/aerodynamics.bdf')

    machs = [0.]
    reduced_freqs = np.linspace(1e-9,1,101)
    write_file = '/MKAERO/MKaero'
    for mi in machs:
        model = BDF(debug=True,log=None)
        mgen.Aerodynamic_model.mkaero(model,mi,reduced_freqs,m3.path+write_file+common.remove_dot(mi)+'.bdf')
################################################################################

################################################################################
# AICs matrices
################################################################################
if run_AICs:
    QsPath=m3.path+'/Qs'
    if not os.path.exists(QsPath):
          os.makedirs(QsPath)    
    FEM_file='/'.join(m3.path.split('/')[:-1]+['FEM'])
    if not os.path.exists(FEM_file):
        os.makedirs(FEM_file)
    femmb.readOP4_matrices(readKM=[m3.path+'/Ka.op4',m3.path+'/Ma.op4'],nameKM=['KAA','MAA'],saveKM=[FEM_file+'/Kaa.npy',FEM_file+'/Maa.npy'])
    femmb.mat2dmig(nids,[m3.path+'/Ka.op4',m3.path+'/Ma.op4'],Mread=['KAA','MAA'],Msave=['KAAX','MAAX'],write_dmig=m3.path+'/dmig.bdf',aset_coord=node_id2position,asetrb={1:[0,1,2,3,5]})
    BeamSeg = gridorder.bdf_order_nodes([[nids[0]-1,nids[-1]-1]],m2.path+'/model103.bdf',nodestart=1)
    gridorder.write_structuralGrid_file(FEM_file+'/structuralGrid.txt',BeamSeg,NumBeams=1)
    for mi in machs:
        mkaero = './MKAERO/MKaero%s.bdf' % common.remove_dot(mi)
        mesh4=BDF(debug=True,log=None)
        m4 = mgen.Model('beam_inpf',mesh4)
        #m4.write_modes('HaleX1',5)
        #pdb.set_trace()
        m4.write_modes('HaleX1x',V.NumModes,1,m4.inp.feminas_dir+'/Models'+'/'+ m4.inp.model+'/variablesrb.py',m4.inp.feminas_dir+'/Models'+'/'+ m4.inp.model)
        am4= mgen.Aerodynamic_model2('aero_inp',m4)
        am4.executiveCD_AICs(V.NumModes,phi0='Phi0.op4',phi1='Phi1.op4')
        am4.caseCD_AICs(add_ccd=1)
        am4.caseCD_AICs(add_ccd=1)
        am4.bulkCD_AICs()
        model='/modelAICs%s.bdf'%common.remove_dot(mi)
        # INCLUDE 'Setup/Dummy_Gust2.bdf'
        m4.write_model(model=model,add_string={'MDLPRM  MLTSPLIN1\n':'BEGIN BULK\n'})
        m4.write_includes(['dmig.bdf'],model=model)
        m4.write_includes(['aerodynamics.bdf'],model=model)
        m4.write_includes([mkaero],model=model)
        m4.write_string(string_find="INCLUDE '%s'" % mkaero + '\n',string_write='ENDDATA',line_find=None,model=model)
        if run_nastran:
            subprocess.call("msc20160 nastran %s scr=yes" %(m4.path+model),shell=True,executable='/bin/bash',cwd=m4.path)
################################################################################


import intrinsic.modesrb as mrb

