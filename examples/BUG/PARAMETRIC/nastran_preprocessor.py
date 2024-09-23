from pyNastran.bdf.bdf import read_bdf
from pyNastran.op2.op2 import read_op2
import numpy as np
import os
import copy
from concurrent.futures import ThreadPoolExecutor,as_completed
from importlib import reload
import multiprocessing
from multiprocessing import Process,Queue
from nastran_tools import read_pch
from bug_param_decoder import *
import pyNastran.op4.op4 as op4
from roger import *
from grad_eig import *
import pickle
import scipy as sp
import shutil
import sys

class NastranPreprocessor:
  def __init__(self,nastran_loc):
    self.nastran_loc=nastran_loc

  def aero_analysis(self):
    bdfmodel=read_bdf(self.bdfname_aero,debug=None)
    if self.params is not None:
      bdfmodel=overwrite_bdf(self.params,bdfmodel=bdfmodel)
    os.makedirs(self.working_dir,exist_ok=True)
    bdfmodel.write_bdf(f'{self.working_dir}/base_aero.bdf')
    change_op4name(f'{self.working_dir}/base_aero.bdf','',self.working_dir,'base')
    command=f'{self.nastran_loc} {self.working_dir}/base_aero.bdf out={self.working_dir} old=no news=no > nul 2>&1'
    os.system(command)
    ahh=process_Q(f'{self.working_dir}/base_qhh.op4')
    ahj=process_Q(f'{self.working_dir}/base_qhj.op4')
    np.save(self.fem_dir+"/ahh.npy",ahh)
    np.save(self.fem_dir+"/ahj.npy",ahj)

  def eigenvalue_analysis(self):
    bdfmodel=read_bdf(self.bdfname,debug=None)
    if self.params is not None:
      bdfmodel=overwrite_bdf(self.params,bdfmodel=bdfmodel)
    os.makedirs(self.working_dir,exist_ok=True)
    change_nummode(self.num_modes_eig,bdfmodel)
    bdfmodel.write_bdf(f'{self.working_dir}/base.bdf')
    modify_cao(f'{self.working_dir}/base.bdf')
    convert_to_pch(f'{self.working_dir}/base.bdf',f'{self.working_dir}/basep.bdf')
    change_nummode(1,bdfname=f'{self.working_dir}/basep.bdf')
    commands=[f'{self.nastran_loc} {self.working_dir}/basep.bdf out={self.working_dir} old=no news=no > nul 2>&1']
    if self.require_nastran_eig:
      commands.append(f'{self.nastran_loc} {self.working_dir}/base.bdf out={self.working_dir} old=no news=no > nul 2>&1')
    #parallel execute command1 and command2
    process_list=[]
    for command in commands:
      process=Process(target=os.system,args=(command,))
      process.start()
      process_list.append(process)
    for process in process_list:
      process.join()
    self.Ka,self.Ma,nid_rom=read_pch(f"{self.working_dir}/basep.pch")
    op2model=read_op2(f"{self.working_dir}/base.op2",debug=None)
    _Ma=shift_mat(self.Ma)
    eigenvalues,eigenvectors_rom=sp.linalg.eigh(self.Ka,_Ma)
    eigenvalues_ns=np.array(op2model.eigenvectors[1].eigns,dtype=np.float64)
    eigenvalues[:len(eigenvalues_ns)]=eigenvalues_ns
    self.eigenvectors=(op2model.eigenvectors[1].data).astype(np.float64)
    nid_full=op2model.eigenvectors[1].node_gridtype[:,0]
    self.id_full2rom=np.where(nid_full==nid_rom[:,None])[1]
    eigenvectors_rom_ns=self.eigenvectors[:,self.id_full2rom].reshape(self.eigenvectors.shape[0],-1).T #(ndim,nmode)
    eigenvectors_rom[:,:eigenvectors_rom_ns.shape[1]]=eigenvectors_rom_ns
    print('done')
    _write_op4modes_mod(self.eigenvectors[:self.num_modes_aero],op4_name=f"{self.working_dir}/Phi.op4")
    self.eigenvalues=eigenvalues
    self.eigenvectors_rom=eigenvectors_rom
    self._save_property()

  def sensitivity_preprocess(self,is_aero=False,require_punch=True):
    keys=self.params.keys()
    base_bdf=read_bdf(self.bdfname,debug=None)
    os.makedirs(self.working_dir,exist_ok=True)
    
    #Write bdfs
    bdfnames=[]
    for key in keys:
      if key[0]=='P':
        decoder=self.params['C'+key[1:]]
        if not decoder.is_variable:
          continue
        scale_param=decoder.scale_param
        process_list=[]
        for i in range(len(self.params[key])*2):
          if self.require_nastran_eig:
            bdfnames.append(f'{self.working_dir}/{key[2:]}_{i}.bdf')
          if require_punch:
            bdfnames.append(f'{self.working_dir}/{key[2:]}p_{i}.bdf')
          process=Process(target=_write_bdf,
                          args=(base_bdf,self.params,i,key,scale_param,self.delta,
                                self.working_dir,is_aero,require_punch))
          process.start()
          process_list.append(process)
        for process in process_list:
          process.join()
    #Run Nastran
    memorymax=0.8/self.num_parallel
    def _run_bdf(bdfname):
      if is_aero:
        command=f'{self.nastran_loc} {bdfname} out={self.working_dir} old=no news=no'
      else:
        command=f'{self.nastran_loc} {bdfname} out={self.working_dir} memorymax={memorymax} old=no news=no'
      os.system(f'{command} > nul 2>&1')
    with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
      futures = []
      for bdfname in bdfnames:
        future=executor.submit(_run_bdf,bdfname)
        futures.append(future)
      for future in as_completed(futures):
        pass
    self.bdfnames=bdfnames

  def sensitivity_preprocess_upwind(self,is_aero=False,require_punch=True,require_op2=True):
    keys=self.params.keys()
    base_bdf=read_bdf(self.bdfname,debug=None)
    os.makedirs(self.working_dir,exist_ok=True)
    
    #Write bdfs
    bdfnames=[]
    for key in keys:
      if key[0]=='P':
        decoder=self.params['C'+key[1:]]
        if not decoder.is_variable:
          continue
        scale_param=decoder.scale_param
        process_list=[]
        for i in range(len(self.params[key])):
          if require_op2:
            bdfnames.append(f'{self.working_dir}/{key[2:]}_{i}.bdf')
          if require_punch:
            bdfnames.append(f'{self.working_dir}/{key[2:]}p_{i}.bdf')
          process=Process(target=_write_bdf_upwind,
                          args=(base_bdf,self.params,i,key,scale_param,self.delta,
                                self.working_dir,is_aero,require_punch))
          process.start()
          process_list.append(process)
        for process in process_list:
          process.join()
    #Run Nastran
    memorymax=0.9/self.num_parallel
    def _run_bdf(bdfname):
      if is_aero:
        command=f'{self.nastran_loc} {bdfname} out={self.working_dir} old=no news=no'
      else:
        command=f'{self.nastran_loc} {bdfname} out={self.working_dir} memorymax={memorymax} old=no news=no'
      os.system(f'{command} > nul 2>&1')
    with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
      futures = []
      for bdfname in bdfnames:
        future=executor.submit(_run_bdf,bdfname)
        futures.append(future)
      for future in as_completed(futures):
        pass

  def sensitivity_structure_fd(self,preprocess=True):
    """
    Calculate derivatives of eigenvalues and eigenvectors using finite difference method
    """
    if preprocess:
      self.sensitivity_preprocess(is_aero=False,)
    #Read output files
    out_dic=dict()
    for key in self.params.keys():
      if key[0]=='P':
        decoder=self.params['C'+key[1:]]
        if not decoder.is_variable:
          continue
        scale_param=decoder.scale_param
        d_Kaa=[]
        d_Maa=[]
        Kaa1_list=[]; Maa1_list=[]; Kaa2_list=[]; Maa2_list=[]
        for i in range(len(self.params[key])):
          Kaa1,Maa1,_=read_pch(f'{self.working_dir}/{key[2:]}p_{i*2}.pch')
          Kaa2,Maa2,_=read_pch(f'{self.working_dir}/{key[2:]}p_{i*2+1}.pch')
          d_Kaa.append((Kaa2-Kaa1)/(2*self.delta*scale_param))
          d_Maa.append((Maa2-Maa1)/(2*self.delta*scale_param))
          Kaa1_list.append(Kaa1); Maa1_list.append(Maa1)
          Kaa2_list.append(Kaa2); Maa2_list.append(Maa2)
        name='_'.join(key.split('_')[2:])
        out_dic['KAA_'+name]=np.array(d_Kaa)
        out_dic['MAA_'+name]=np.array(d_Maa)

        #calculate sensitivity of eigenvalues and eigenvectors
        d_evec=[]
        d_eval=[]
        for i in range(len(self.params[key])):
          if self.require_nastran_eig: #use nastran eigenvalue analysis results
            op2_1=read_op2(f'{self.working_dir}/{key[2:]}_{i*2}.op2',debug=None)
            evec1=op2_1.eigenvectors[1].data.astype(np.float64)
            evec1=evec1[:,self.id_full2rom].reshape(evec1.shape[0],-1).T #(ndim,nmode)
            eval1=np.array(op2_1.eigenvectors[1].eigns)
            op2_2=read_op2(f'{self.working_dir}/{key[2:]}_{i*2+1}.op2',debug=None)
            evec2=op2_2.eigenvectors[1].data.astype(np.float64)
            evec2=evec2[:,self.id_full2rom].reshape(evec2.shape[0],-1).T #(ndim,nmode)
            eval2=np.array(op2_2.eigenvectors[1].eigns)
          else: #use python eigenvalue analysis results
            Kaa1=Kaa1_list[i]; Maa1=Maa1_list[i]
            Kaa2=Kaa2_list[i]; Maa2=Maa2_list[i]
            eval1,evec1=sp.linalg.eigh(Kaa1,Maa1)
            eval2,evec2=sp.linalg.eigh(Kaa2,Maa2)
          evec1,eval1=rectify_eigs(self.eigenvectors_rom,evec1,eval1)
          evec2,eval2=rectify_eigs(self.eigenvectors_rom,evec2,eval2)
          d_evec.append((evec2-evec1)/(2*self.delta*scale_param))
          d_eval.append((eval2-eval1)/(2*self.delta*scale_param))
        name='_'.join(key.split('_')[2:])
        out_dic['EVEC_'+name]=np.array(d_evec)
        out_dic['EVAL_'+name]=np.array(d_eval)
    return out_dic
  
  def sensitivity_structure(self,):
    if self.require_preprocess:
      self.sensitivity_preprocess(is_aero=False,require_punch=True,require_op2=False)
    #Read output files
    out_dic=dict()
    
    for key in self.params.keys():
      if key[0]=='P':
        decoder=self.params['C'+key[1:]]
        if not decoder.is_variable:
          continue
        scale_param=decoder.scale_param
        
        d_Kaa=[]
        d_Maa=[]
        for i in range(len(self.params[key])):
          Kaa1,Maa1,_=read_pch(f'{self.working_dir}/{key[2:]}p_{i*2}.pch')
          Kaa2,Maa2,_=read_pch(f'{self.working_dir}/{key[2:]}p_{i*2+1}.pch')
          d_Kaa.append((Kaa2-Kaa1)/(2*self.delta*scale_param))
          d_Maa.append((Maa2-Maa1)/(2*self.delta*scale_param))
        name='_'.join(key.split('_')[2:])
        d_Kaa=np.array(d_Kaa)
        d_Maa=np.array(d_Maa)
        out_dic['KAA_'+name]=d_Kaa
        out_dic['MAA_'+name]=d_Maa
        #calculate sensitivity of eigenvalues and eigenvectors
        d_eval,d_evec=grad_eig(self.Ka,self.Ma,self.eigenvalues,self.eigenvectors_rom,d_Kaa,d_Maa)
        out_dic['EVEC_'+name]=d_evec
        out_dic['EVAL_'+name]=d_eval
    with open(self.fem_dir+"/sensitivity_nastran.pkl","wb") as f:
      pickle.dump(out_dic, f)
    self.grad_param=out_dic
    return out_dic
  
  def sensitivity_structure_upwind(self):
    if self.require_preprocess:
      self.sensitivity_preprocess_upwind(is_aero=False,require_punch=True,require_op2=False)
    #Read output files
    out_dic=dict()
    for key in self.params.keys():
      if key[0]=='P':
        decoder=self.params['C'+key[1:]]
        if not decoder.is_variable:
          continue
        scale_param=decoder.scale_param
        d_Kaa=[]
        d_Maa=[]
        for i in range(len(self.params[key])):
          Kaa,Maa,_=read_pch(f'{self.working_dir}/{key[2:]}p_{i}.pch')
          d_Kaa.append((Kaa-self.Ka)/(self.delta*scale_param))
          d_Maa.append((Maa-self.Ma)/(self.delta*scale_param))
        name='_'.join(key.split('_')[2:])
        d_Kaa=np.array(d_Kaa)
        d_Maa=np.array(d_Maa)
        out_dic['KAA_'+name]=d_Kaa
        out_dic['MAA_'+name]=d_Maa
        #calculate sensitivity of eigenvalues and eigenvectors
        d_eval,d_evec=grad_eig(self.Ka,self.Ma,self.eigenvalues,self.eigenvectors_rom,d_Kaa,d_Maa)
        out_dic['EVEC_'+name]=d_evec
        out_dic['EVAL_'+name]=d_eval
    with open(self.fem_dir+"/sensitivity_nastran.pkl","wb") as f:
      pickle.dump(out_dic, f)
    self.grad_param=out_dic
    return out_dic

  def sensitivity_aero(self,params:dict,bdfname,delta=1e-3,num_parallel=4,working_dir='./_temp/',require_preprocess=True):
    if require_preprocess:
      self.sensitivity_preprocess(params,bdfname,delta,num_parallel,is_aero=True,
                                  require_punch=False,working_dir=working_dir)
    out_dic=dict()
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
  
  def set_config(self,bdfname,bdfname_aero=None,working_dir='./_temp',delta=1e-2,
                 num_parallel=4,require_preprocess=True,fem_dir='./FEM',
                 num_modes_aero=50,num_modes_eig=50,require_nastran_eig=True):
    self.bdfname=bdfname
    self.bdfname_aero=bdfname_aero
    self.working_dir=working_dir
    self.delta=delta
    self.num_parallel=num_parallel
    self.require_preprocess=require_preprocess
    self.fem_dir=fem_dir
    self.num_modes_aero=num_modes_aero
    self.num_modes_eig=num_modes_eig
    self.require_nastran_eig=require_nastran_eig
    os.makedirs(self.working_dir,exist_ok=True)
    os.makedirs(self.fem_dir,exist_ok=True)

  
  def set_params(self,params):
    self.params=params

  def _save_property(self):
    os.makedirs(self.fem_dir,exist_ok=True)
    np.save(f'{self.fem_dir}/Ka.npy',self.Ka.astype(np.float64))
    np.save(f'{self.fem_dir}/Ma.npy',self.Ma.astype(np.float64))
    np.save(f'{self.fem_dir}/eigenvals.npy',self.eigenvalues.astype(np.float64))
    np.save(f'{self.fem_dir}/eigenvecs.npy',self.eigenvectors_rom.astype(np.float64))

  def parametric_study(self,params:list[dict],fname_grid,fname_feniax,aero=False,):
    """
    Execute multiple parameter inputs
    """
    self.fname_feniax=fname_feniax
    if aero:
      raise NotImplementedError
    self.parametric_dir=self.working_dir+'/parametric'
    os.makedirs(self.parametric_dir,exist_ok=True)
    #write bdf files
    process_list=[]
    bdfnames=[]
    dirnames=[]
    for i,param in enumerate(params):
      dir_name=f'{self.parametric_dir}/case{i}'
      os.makedirs(dir_name,exist_ok=True)
      os.makedirs(dir_name+'/nastran_files',exist_ok=True)
      with open(f'{self.parametric_dir}/case{i}/param.pkl','wb') as f:
        pickle.dump(param,f)
      process=Process(target=_write_bdf_p,args=(self.bdfname,param,dir_name+'/nastran_files',True))
      bdfnames.append(f'{dir_name}/nastran_files/main.bdf')
      bdfnames.append(f'{dir_name}/nastran_files/main_p.bdf')
      dirnames.append(dir_name+'/nastran_files');dirnames.append(dir_name+'/nastran_files')
      process.start()
      process_list.append(process)
    for process in process_list:
      process.join()
    #run nastran
    memorymax=0.8/self.num_parallel
    def _run_bdf(bdfname,out_dir):
      command=f'{self.nastran_loc} {bdfname} out={out_dir} memorymax={memorymax} old=no news=no'
      os.system(f'{command} > nul 2>&1')
    with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
      futures = []
      for bdfname,dirname in zip(bdfnames,dirnames):
        future=executor.submit(_run_bdf,bdfname,dirname)
        futures.append(future)
      for future in as_completed(futures):
        pass
    self.bdfnames=bdfnames
    #read output files

    for i in range(len(params)):
      dir_name=f'{self.parametric_dir}/case{i}'
      os.makedirs(dir_name+'/FEM',exist_ok=True)
      Ka,Ma,nid_rom=read_pch(f"{dir_name}/nastran_files/main_p.pch")
      op2model=read_op2(f"{dir_name}/nastran_files/main.op2",debug=None)
      _Ma=shift_mat(Ma)
      eigenvalues,eigenvectors_rom=sp.linalg.eigh(Ka,_Ma)
      eigenvalues_ns=np.array(op2model.eigenvectors[1].eigns,dtype=np.float64)
      eigenvalues[:len(eigenvalues_ns)]=eigenvalues_ns
      eigenvectors=(op2model.eigenvectors[1].data).astype(np.float64)
      nid_full=op2model.eigenvectors[1].node_gridtype[:,0]
      id_full2rom=np.where(nid_full==nid_rom[:,None])[1]
      eigenvectors_rom_ns=eigenvectors[:,id_full2rom].reshape(eigenvectors.shape[0],-1).T #(ndim,nmode)
      eigenvectors_rom[:,:eigenvectors_rom_ns.shape[1]]=eigenvectors_rom_ns
      np.save(f'{dir_name}/FEM/Ka.npy',Ka)
      np.save(f'{dir_name}/FEM/Ma.npy',Ma)
      np.save(f'{dir_name}/FEM/eigenvals.npy',eigenvalues)
      np.save(f'{dir_name}/FEM/eigenvecs.npy',eigenvectors_rom)
      shutil.copy(fname_grid,f'{dir_name}/FEM/structuralGrid')

  def _parametric_study_feniax(self,params):
    abs_path=os.path.abspath(self.fname_feniax)
    path_settings=os.path.dirname(abs_path)
    fname_setting=os.path.basename(abs_path)
    sys.path.append(path_settings)
    cwd=os.getcwd()
    for i,param in enumerate(params):
      os.chdir(f'{self.parametric_dir}/case{i}')
      try:
        exec(f'reload({fname_setting})')
      except NameError:
        exec(f'import {fname_setting}')
      os.chdir(cwd)

def overwrite_bdf(params:dict,bdfname=None,bdfmodel=None):
  if bdfmodel is None:
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
  change_nummode(1,bdfname=fname_out)

def change_op4name(fname,idx,working_dir,label):
  f=open(fname,'r')
  lines=f.readlines()
  f.close()
  n=0
  for i,line in enumerate(lines):
    if "assign OUTPUT4='./data_out/Qhh" in line:
      lines[i]=f"assign OUTPUT4='{working_dir}/{label}_Qhh{idx}.op4',formatted,UNIT=11\n"
      n+=1
    if "assign OUTPUT4='./data_out/Qhj" in line:
      lines[i]=f"assign OUTPUT4='{working_dir}/{label}_Qhj{idx}.op4',formatted,UNIT=12\n"
      n+=1
    if "assign INPUTT4='./data_out/Phi" in line:
      lines[i]=f"assign INPUTT4='{working_dir}/Phi.op4',formatted,UNIT=90\n"
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
  _bdfmodel.write_bdf(f'{working_dir}/{key[2:]}_{idx}.bdf')
  if is_aero:
    change_op4name(f'{working_dir}/{key[2:]}_{idx}.bdf',idx,working_dir,key[2:])
  if require_punch:
    convert_to_pch(f'{working_dir}/{key[2:]}_{idx}.bdf',
                  f'{working_dir}/{key[2:]}p_{idx}.bdf')
    
def _write_bdf_upwind(bdfmodel,params,idx,key,scale_param,delta,working_dir,is_aero,require_punch):
  _params=copy.deepcopy(params)
  _params[key][idx]=params[key][idx]+delta*scale_param
  _bdfmodel=overwrite_bdf(_params,bdfmodel=copy.copy(bdfmodel))
  _bdfmodel.write_bdf(f'{working_dir}/{key[2:]}_{idx}.bdf')
  if is_aero:
    change_op4name(f'{working_dir}/{key[2:]}_{idx}.bdf',idx,working_dir,key[2:])
  if require_punch:
    convert_to_pch(f'{working_dir}/{key[2:]}_{idx}.bdf',
                  f'{working_dir}/{key[2:]}p_{idx}.bdf')

def _write_bdf_p(bdfname,params,dir_name,require_punch):
  bdfmodel=overwrite_bdf(params,bdfmodel=read_bdf(bdfname,debug=None))
  bdfmodel.write_bdf(f'{dir_name}/main.bdf')
  if require_punch:
    convert_to_pch(f'{dir_name}/main.bdf',f'{dir_name}/main_p.bdf')

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

def _write_op4modes_mod(eigvecs,op4_name: None,matrix_name='PHG'):
  nummodes, numnodes, numdim = eigvecs.shape
  eigvecs = eigvecs.reshape((nummodes,numnodes*numdim)).T
  op4_data = op4.OP4(debug=None)
  op4_data.write_op4(op4_name,{matrix_name:(2,eigvecs)},is_binary=False)

def find_close_vec(v_trg,v_ref):
  """
  v_trg: (ndim,nmodeT)
  v_ref: (ndim,nmodeR)
  """
  norm_trg=np.linalg.norm(v_trg,axis=0) #(nmodeT,)
  norm_ref=np.linalg.norm(v_ref,axis=0) #(nmodeR,)
  dot=v_trg.T@v_ref #(nmodeT,nmodeR)
  cossim=dot/norm_trg[:,None]/norm_ref[None,:] #(nmodeT,nmodeR)
  #print(cossim.shape)
  idx=np.argmax(np.abs(cossim),axis=1)
  sign=np.sign(cossim[np.arange(cossim.shape[0]),idx])
  return idx,sign

def rectify_eigs(v_trg,v_ref,l_ref):
  """
  v_trg: (ndim,nmodeT)
  v_ref: (ndim,nmodeR)
  l_ref: (nmodeR,)
  """
  idx,sign=find_close_vec(v_trg,v_ref)
  v_refM=v_ref[:,idx]*sign
  l_refM=l_ref[idx]
  return v_refM,l_refM

def change_nummode(num_modes,bdfmodel=None,bdfname=None):
  if bdfmodel is None:
    bdfmodel=read_bdf(bdfname,debug=None)
  for idx in bdfmodel.methods:
    if bdfmodel.methods[idx].type=='EIGRL':
      bdfmodel.methods[idx].nd=num_modes
      break
  if bdfname is not None:
    bdfmodel.write_bdf(bdfname)