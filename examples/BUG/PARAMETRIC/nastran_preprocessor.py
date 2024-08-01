from pyNastran.bdf.bdf import read_bdf
from pyNastran.op2.op2 import read_op2
import numpy as np
import os
import copy
from concurrent.futures import ThreadPoolExecutor,as_completed
from multiprocessing import Process
from nastran_tools import read_pch
from bug_param_decoder import *
import pyNastran.op4.op4 as op4
from roger import *

class NastranPreprocessor:
  def __init__(self,nastran_loc):
    self.nastran_loc=nastran_loc

  def sensitivity_preprocess(self,params:dict,bdfname,delta=1e-3,num_parallel=4,
                             is_aero=False,require_punch=True,working_dir='./_temp/'):
    
    keys=params.keys()
    base_bdf=read_bdf(bdfname,debug=None)
    os.makedirs(working_dir,exist_ok=True)
    
    #Write bdfs
    bdfnames=[]
    for key in keys:
      if key[0]=='P':
        decoder=params['C'+key[1:]]
        if not decoder.is_variable:
          continue
        scale_param=decoder.scale_param
        process_list=[]
        for i in range(len(params[key])*2):
          bdfnames.append(f'{working_dir}{key[2:]}_{i}.bdf')
          if require_punch:
            bdfnames.append(f'{working_dir}{key[2:]}p_{i}.bdf')
          process=Process(target=_write_bdf,
                          args=(base_bdf,params,i,key,scale_param,delta,
                                working_dir,is_aero,require_punch))
          process.start()
          process_list.append(process)
        for process in process_list:
          process.join()
    #Run Nastran
    memorymax=0.8/num_parallel
    def _run_bdf(bdfname):
      if is_aero:
        command=f'{self.nastran_loc} {bdfname} out={working_dir} old=no news=no'
      else:
        command=f'{self.nastran_loc} {bdfname} out={working_dir} memorymax={memorymax} old=no news=no'
      os.system(f'{command} > nul 2>&1')
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
      futures = []
      for bdfname in bdfnames:
        future=executor.submit(_run_bdf,bdfname)
        futures.append(future)
      for future in as_completed(futures):
        pass

  def sensitivity_structure(self,params:dict,bdfname,delta=1e-3,num_parallel=4,require_punch=False,working_dir='./_temp/'):
    self.sensitivity_preprocess(params,bdfname,delta,num_parallel,is_aero=False,require_punch=require_punch,working_dir=working_dir)
    #Read output files
    out_dic=copy.deepcopy(params)
    for key in params.keys():
      if key[0]=='P':
        decoder=params['C'+key[1:]]
        if not decoder.is_variable:
          continue
        scale_param=decoder.scale_param
        d_evec=[]
        d_eval=[]
        for i in range(len(params[key])):
          op2_1=read_op2(f'{working_dir}{key[2:]}_{i*2}.op2',debug=None)
          op2_2=read_op2(f'{working_dir}{key[2:]}_{i*2+1}.op2',debug=None)
          d_evec.append((op2_2.eigenvectors[1].data-op2_1.eigenvectors[1].data)/(2*delta*scale_param))
          d_eval.append((np.array(op2_2.eigenvectors[1].eigns)-np.array(op2_1.eigenvectors[1].eigns))/(2*delta*scale_param))
        out_dic['S'+key[1:]+'_EVEC']=np.array(d_evec)
        out_dic['S'+key[1:]+'_EVAL']=np.array(d_eval)
        if require_punch:
          d_Kaa=[]
          d_Maa=[]
          for i in range(len(params[key])):
            Kaa1,Maa1=read_pch(f'{working_dir}{key[2:]}p_{i*2}.pch')
            Kaa2,Maa2=read_pch(f'{working_dir}{key[2:]}p_{i*2+1}.pch')
            d_Kaa.append((Kaa2-Kaa1)/(2*delta*scale_param))
            d_Maa.append((Maa2-Maa1)/(2*delta*scale_param))
          out_dic['S'+key[1:]+'_KAA']=np.array(d_Kaa)
          out_dic['S'+key[1:]+'_MAA']=np.array(d_Maa)

    return out_dic

  def sensitivity_aero(self,params:dict,bdfname,delta=1e-3,num_parallel=4,working_dir='./_temp/',require_preprocess=True):
    if require_preprocess:
      self.sensitivity_preprocess(params,bdfname,delta,num_parallel,is_aero=True,
                                  require_punch=False,working_dir=working_dir)
    out_dic=copy.deepcopy(params)
    for key in params.keys():
      if key[0]=='P':
        decoder=params['C'+key[1:]]
        if not decoder.is_variable:
          continue
        if not 'CAERO' in key: #only process caero parameters
          continue
        scale_param=decoder.scale_param
        d_ahh=[]
        d_ahj=[]
        for i in range(len(params[key])):
          qhh_1=process_Q(f'{working_dir}{key[2:]}_qhh{i*2}.op4')
          qhh_2=process_Q(f'{working_dir}{key[2:]}_qhh{i*2+1}.op4')
          qhj_1=process_Q(f'{working_dir}{key[2:]}_qhj{i*2}.op4')
          qhj_2=process_Q(f'{working_dir}{key[2:]}_qhj{i*2+1}.op4')
          d_ahh.append((qhh_2-qhh_1)/(2*delta*scale_param))
          d_ahj.append((qhj_2-qhj_1)/(2*delta*scale_param))
        out_dic['S'+key[1:]+'_AHH']=np.array(d_ahh)
        out_dic['S'+key[1:]+'_AHJ']=np.array(d_ahj)
    return out_dic
        
  def eigenvalue_analysis(self,params:dict|None,bdfname,working_dir='./_temp/'):
    bdfmodel=read_bdf(bdfname,debug=None)
    if params is not None:
      bdfmodel=overwrite_bdf(params,bdfmodel=bdfmodel)
    os.makedirs(working_dir,exist_ok=True)
    bdfmodel.write_bdf(f'{working_dir}base.bdf')
    modify_cao(f'{working_dir}base.bdf')
    convert_to_pch(f'{working_dir}base.bdf',f'{working_dir}basep.bdf')
    command1=f'{self.nastran_loc} {working_dir}base.bdf out={working_dir} memorymax=0.8 old=no news=no > nul 2>&1'
    command2=f'{self.nastran_loc} {working_dir}basep.bdf out={working_dir} memorymax=0.8 old=no news=no > nul 2>&1'
    #parallel execute command1 and command2
    with ThreadPoolExecutor(max_workers=2) as executor:
      futures = []
      for command in [command1,command2]:
        future=executor.submit(os.system,command)
        futures.append(future)
      for future in as_completed(futures):
        pass
    num_modes=50
    _write_op4modes(f"{working_dir}base.op2",num_modes,op4_name=f"{working_dir}Phi.op4")
    self.Ka,self.Ma,nid_rom=read_pch(f"{working_dir}basep.pch")
    op2model=read_op2(f"{working_dir}base.op2",debug=None)
    self.eigenvalues=np.array(op2model.eigenvectors[1].eigns)
    self.eigenvectors=op2model.eigenvectors[1].data
    nid_full=op2model.eigenvectors[1].node_gridtype[:,0]
    self.id_full2rom=np.where(nid_full==nid_rom[:,None])[1]
    self.eigenvectors_rom=self.eigenvectors[:,self.id_full2rom].reshape(self.eigenvectors.shape[0],-1).T

  def save_property(self,fem_dir):
    os.makedirs(fem_dir,exist_ok=True)
    np.save(f'{fem_dir}/Ka.npy',self.Ka)
    np.save(f'{fem_dir}/Ma.npy',self.Ma)
    np.save(f'{fem_dir}/eigenvals.npy',self.eigenvalues)
    np.save(f'{fem_dir}/eigenvecs.npy',self.eigenvectors)


def overwrite_bdf(params:dict,bdfname=None,bdfmodel=None):
  if bdfmodel is None:
    bdfmodel=read_bdf(bdfname,debug=None)
  keys=params.keys()
  for key in keys:
    if key[:3]=='P_P': #Property entry
      #decode parameter
      val=params['C_'+key[1:]].get_val(params[key])
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

#def run_bdfmodel(bdfmodel,bdfname,working_dir,memorymax=0.3):
#  bdfmodel.write_bdf(bdfname)
#  command=f'{NASTRAN_LOC} {bdfname} out={working_dir} memorymax={memorymax} old=no news=no'
#  os.system(f'{command} > nul 2>&1')
  
def convert_to_pch(fname_in,fname_out):
  f=open(fname_in,'r')
  lines=f.readlines()
  f.close()
  n=0
  for i,line in enumerate(lines):
    if 'DISP' in line:
      lines[i]='$'+line
      n+=1
    if 'BEGIN BULK' in line:
      lines.insert(i+1,'PARAM,EXTOUT,DMIGPCH\n')
      n+=1
    if n==2:
      break
  f=open(fname_out,'w')
  f.writelines(lines)
  f.close()

def change_op4name(fname,idx,working_dir,label):
  f=open(fname,'r')
  lines=f.readlines()
  f.close()
  n=0
  for i,line in enumerate(lines):
    if "assign OUTPUT4='./data_out/Qhh" in line:
      lines[i]=f"assign OUTPUT4='{working_dir}{label}_Qhh{idx}.op4',formatted,UNIT=11\n"
      n+=1
    if "assign OUTPUT4='./data_out/Qhj" in line:
      lines[i]=f"assign OUTPUT4='{working_dir}{label}_Qhj{idx}.op4',formatted,UNIT=12\n"
      n+=1
    if "assign INPUTT4='./data_out/Phi" in line:
      lines[i]=f"assign INPUTT4='{working_dir}Phi.op4',formatted,UNIT=90\n"
      n+=1
    if n==3:
      break
  f=open(fname,'w')
  f.writelines(lines)
  f.close()

def modify_cao(fname):
  f=open(fname,'r')
  lines=f.readlines()
  f.close()
  n=0
  for i,line in enumerate(lines):
    if 'SPCF = ALL' in line:
      lines[i]='$'+line
      n+=1
    if 'DISP' in line:
      lines[i]='DISPLACEMENT(PLOT)=ALL\n'
      n+=1
    if n==2:
      break
      
  f=open(fname,'w')
  f.writelines(lines)
  f.close()

def _write_bdf(bdfmodel,params,idx,key,scale_param,delta,working_dir,is_aero,require_punch):
  i=idx//2
  sgn=1.0 if (idx%2==1) else -1.0
  _params=copy.deepcopy(params)
  _params[key][i]=params[key][i]+delta*sgn*scale_param
  _bdfmodel=overwrite_bdf(_params,bdfmodel=copy.copy(bdfmodel))
  _bdfmodel.write_bdf(f'{working_dir}{key[2:]}_{idx}.bdf')
  if is_aero:
    change_op4name(f'{working_dir}{key[2:]}_{idx}.bdf',idx,working_dir,key[2:])
  if require_punch:
    convert_to_pch(f'{working_dir}{key[2:]}_{idx}.bdf',
                  f'{working_dir}{key[2:]}p_{idx}.bdf')
    
def save_params(params:dict,dir:str):
  for key in params.keys():
    if key[0]=='P': #save parameter
      np.save(f'{dir}/{key}.npy',params[key])
      np.save(f'{dir}/I{key[1:]}.npy',params['C'+key[1:]].idx)
      if params['C'+key[1:]].coord_control is not None:
        np.save(f'{dir}/V{key[1:]}.npy',params['C'+key[1:]].coord_control) 
    elif key[0]=='S': #save sensitivity
      np.save(f'{dir}/{key}.npy',params[key])

def load_params(dir:str,handler):
  params=dict()
  #get file names in the directory that ends with .npy
  fnames=[f for f in os.listdir(dir) if f.endswith('.npy')]
  for fname in fnames:
    if fname[:2]=='P_':
      key=fname[:-4]
      params[key]=np.load(f'{dir}/{fname}')
      idx=np.load(f'{dir}/I{key[1:]}.npy')
      if os.path.exists(f'{dir}/V{key[1:]}.npy'):
        coord_control=np.load(f'{dir}/V{key[1:]}.npy')
      else:
        coord_control=None
      #exec(f'decoder={key.split('_')[1]}(idx,handler,coord_control)')
      decoder=eval(f'{key.split("_")[1]}')(idx,handler,coord_control)
      params['C'+key[1:]]=decoder
    elif fname[:2]=='S_':
      key=fname[:-4]
      params[key]=np.load(f'{dir}/{fname}')
  return params

def _write_op4modes(op2_name:str,num_modes: int,op4_name: None,
                   matrix_name='PHG'):
    op2=read_op2(op2_name,debug=None)
    eig1 = op2.eigenvectors[1]
    modesdata = eig1.data
    modes = modesdata[:num_modes]
    op2_nummodes, op2.numnodes, op2.numdim = modes.shape
    modes_reshape = modes.reshape((op2_nummodes, op2.numnodes * op2.numdim)).T
    op4_data = op4.OP4(debug=None)
    op4_data.write_op4(op4_name,{matrix_name:(2,modes_reshape)},is_binary=False)