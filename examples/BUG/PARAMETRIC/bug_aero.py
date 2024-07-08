from pyNastran.bdf.bdf import read_bdf
from fem4inas.unastran.aero import GenDLMPanels
import PostProcessing.panels as panels
import fem4inas.plotools.nastranvtk.bdfdef as bdfdef
import numpy as np
import fem4inas.unastran.aero as nasaero
import pyNastran.op4.op4 as op4
import pathlib


def generate_dlm(bdf_input='./BUGaero1.bdf',bdf_output='./dlm_model.bdf',
                paraview_dir='./dlm_mesh',bdf_structure='./BUG_103cao.bdf',
                write_paraview=False):
  bdfaero=read_bdf(bdf_input,debug=None,validate=False, punch=False)

  ######## BUILD AERO MODEL ##############
  aeros = dict(RWing1=dict(nspan=2, nchord=8),
              RWing2=dict(nspan=3, nchord=8),
              RWing3=dict(nspan=9, nchord=8),
              RWing4=dict(nspan=6, nchord=8),
              RWing5=dict(nspan=4, nchord=8),
              RHTP=dict(nspan=6, nchord=8),
              LWing1=dict(nspan=2, nchord=8),
              LWing2=dict(nspan=3, nchord=8),
              LWing3=dict(nspan=9, nchord=8),
              LWing4=dict(nspan=6, nchord=8),
              LWing5=dict(nspan=4, nchord=8),
              LHTP=dict(nspan=6, nchord=8),)

  aeros2ids = dict(RWing1=3504001,
                  RWing2=3500001,
                  RWing3=3501001,
                  RWing4=3502001,
                  RWing5=3503001,
                  RHTP=3600001)

  for ki, i in aeros2ids.items():
      aeros[ki]['p1'] = bdfaero.caeros[i].p1
      aeros[ki]['p4'] = bdfaero.caeros[i].p4
      aeros[ki]['x12'] = bdfaero.caeros[i].x12
      aeros[ki]['x43'] = bdfaero.caeros[i].x43
      ki_l=('L'+ki[1:])
      aeros[ki_l]['p1'] = bdfaero.caeros[i].p1*np.array([1.,-1.,1.])
      aeros[ki_l]['p4'] = bdfaero.caeros[i].p4*np.array([1.,-1.,1.])
      aeros[ki_l]['x12'] = bdfaero.caeros[i].x12
      aeros[ki_l]['x43'] = bdfaero.caeros[i].x43

  aeros['RWing1']['set1x'] = [1004, 2001] 
  aeros['RWing2']['set1x'] = [2003, 2005, 2008, 2010] 
  aeros['RWing3']['set1x'] = list(range(2012, 2030, 2))
  aeros['RWing4']['set1x'] = list(range(2030, 2044, 2))
  aeros['RWing5']['set1x'] = list(range(2044,2053, 2))
  aeros['RHTP']['set1x'] = list(range(4000, 4014))

  aeros['LWing1']['set1x'] = [1004, 10002001] 
  aeros['LWing2']['set1x'] = [10002003, 10002005, 10002008, 100020010] 
  aeros['LWing3']['set1x'] = list(range(10002012, 10002030, 2))
  aeros['LWing4']['set1x'] = list(range(10002030, 10002044, 2))
  aeros['LWing5']['set1x'] = list(range(10002044,10002053, 2))
  aeros['LHTP']['set1x'] = [4000]+list(range(10004001, 10004014))

  dlm = GenDLMPanels.from_dict(aeros) # pass your dictionary with DLM model
  dlm.build_model()
  dlm.model.write_bdf(bdf_output) # write the bdf file
  if write_paraview:
    grid = panels.caero2grid(dlm.components, dlm.caero1) # build grid from dlm model
    panels.build_gridmesh(grid,paraview_dir)  #  write paraview mesh
    bdfdef.vtkRef(bdf_structure)  # write full FE paraview

def generate_aero(modesdata,eigsdata,num_modes=50,mach=0.8,u_inf=200.,rho_inf=1.5):
  machs=[mach]
  reduced_freqs = np.hstack([1e-6, np.linspace(1e-3,1, 50)]) #np.hstack([np.linspace(1e-5,1, 50), [10-0.001, 10., 10+0.001]])
  
  #chord_panels = dict(wing=15, hstabilizer=10, vstabilizer=10)
  #aero['s_ref'] = 361.6
  #aero['b_ref'] = 58.0
  #aero['X_ref'] = 36.3495
  flutter_id = 9010
  mach_fact = machs
  kv_fact = [200., 220.]
  density_fact = [rho_inf]
  c_ref = 1.
  #b_ref = 28.8*2
  #S_ref = b_ref * c_ref
  rho_ref=rho_inf
  #q_inf = 0.5 * rho_inf * u_inf ** 2
  #alpha_sol144 = 1 * np.pi / 180
  flutter_method="PK"
  flutter_sett = dict()
  aero_sett = dict()
  #num_poles = 5
  #gst_lengths = [18.0,42.,67.,91.,116.,140.,165.0,189.,214.]
  
  #from fem4inas.utils import write_op4modes
  write_op4modes(modesdata,eigsdata,num_modes,
                  op4_name=f"./data_out/Phi{num_modes}",return_modes=False)

  dlm_gafs = nasaero.GenFlutter(flutter_id,
                                density_fact,
                                mach_fact,
                                kv_fact,
                                machs,
                                reduced_freqs,
                                u_inf,
                                c_ref,
                                rho_ref,
                                flutter_method,
                                flutter_sett,
                                aero_sett)
  dlm_gafs.build_model()
  dlm_gafs.model.write_bdf("./GAFs/aero_flutter.bdf")

def write_op4modes(modesdata,eigsdata,num_modes: int,op4_name: None,
                   matrix_name='PHG',return_modes=False):
  modes = modesdata[:num_modes]
  op2_nummodes, numnodes, numdim = modes.shape
  modes_reshape = modes.reshape((op2_nummodes, numnodes * numdim)).T
  op4_data = op4.OP4()
  op4_name = str(pathlib.Path(op4_name).with_suffix('.op4'))
  op4_data.write_op4(op4_name,{matrix_name:(2, modes_reshape)},is_binary=False)
  if return_modes:
      return eigsdata, modesdata