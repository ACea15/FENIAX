from pyNastran.bdf.bdf import read_bdf
import numpy as np
import os

NASTRAN_LOC=None

class NastranPreprocessor:
  def __init__(self,bdfname):
    self.bdfname=bdfname
  
  def load_input(self,params:dict):
    self.bdf=read_bdf(self.bdfname,debug=None)
    self.params=params

  def sensitivity_strucural(self,params:dict,bdfname,delta=1e-3):
    if NASTRAN_LOC is None:
      raise ValueError('NASTRAN_LOC not set')
    bdf_list=[]
    keys=params.keys()
    for key in keys:
      if (key[:3]=='P_P') or (key[:6]=='P_CONM') or key[:5]=='P_MAT': #Property entry
        param_ref=params[key]
        for i in range(len(params[key])):
          param_temp=params.copy()
          param_temp[key][i]=param_ref[i]-delta
          bdf_list.append(overwrite_bdf(param_temp,bdfname))
          param_temp=params.copy()
          param_temp[key][i]=param_ref[i]+delta
          bdf_list.append(overwrite_bdf(param_temp,bdfname))

  
  def sensitivity_aero(self,params:dict,bdfname):
    pass

def overwrite_bdf(params:dict,bdfname):
  bdfmodel=read_bdf(bdfname,debug=None)
  keys=params.keys()
  for key in keys:
    if key[:3]=='P_P': #Property entry
      #decode parameter
      val=params['C'+key[1:]].get_val(params[key])
      properties=list(val.keys())
      properties.remove('pid')
      #overwrite Property values
      for property in properties:
        for i,pid in enumerate(val['pid']):
          exec(f'bdfmodel.properties[pid].{property}=val[property][i]')

    elif key[:6]=='P_CONM':
      #decode parameter
      val=params['C'+key[1:]].get_val(params[key])
      properties=list(val.keys())
      properties.remove('eid')
      #overwrite CONM values
      for property in properties:
        for i,eid in enumerate(val['eid']):
          exec(f'bdfmodel.masses[eid].{property}=val[property][i]')
    
    elif key[:5]=='P_MAT':
      #decode parameter
      val=params['C'+key[1:]].get_val(params[key])
      properties=list(val.keys())
      properties.remove('mid')
      #overwrite MAT values
      for property in properties:
        for i,mid in enumerate(val['mid']):
          exec(f'bdfmodel.materials[mid].{property}=val[property][i]')

    elif key[:5]=='P_CAE':
      #decode parameter
      val=params['C'+key[1:]].get_val(params[key])
      properties=list(val.keys())
      properties.remove('eid')
      #overwrite CAERO values
      for property in properties:
        for i,eid in enumerate(val['eid']):
          exec(f'bdfmodel.caeros[eid].{property}=val[property][i]')
  
  return bdfmodel

def run_bdfmodel(bdfmodel,bdfname,working_dir,memorymax=0.3):
  bdfmodel.write_bdf(bdfname)
  command=f'{NASTRAN_LOC} {bdfname} out={working_dir} memorymax={memorymax} old=no news=no'
  os.system(f'{command} > nul 2>&1')
  