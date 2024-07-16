import numpy as np
import scipy as sp
import os
import concurrent.futures
from pyNastran.op2.op2 import read_op2
from pyNastran.op4.op4 import read_op4
from pch_parser import read_pch
from bug_aero import *
from roger import *
import jax.numpy as jnp

NASTRAN_LOC='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'

class NastranDesignModel:
  def __init__(self,bdf_name):
    self.bdf_name=bdf_name

  def return_bdf(params,fname,*args):
    pass

  def return_bdfs(params,fnames,*args):
    pass

class NastranHandler:
  """
  param : (nparam,)
  """
  #def __init__(self,model:NastranDesignModel,dir_input:str,dir_output:str):
  def __init__(self,cls_NatranModel:type(NastranDesignModel),bdfname,dir_input:str,dir_output:str):
    self.cls_NatranModel=cls_NatranModel
    self.model=cls_NatranModel(bdfname)
    self.dir_input=dir_input
    self.dir_output=dir_output
  
  def _run_nastran(self,param,*args):
    fname=f'{self.dir_input}rom.bdf'
    self.model.return_bdf(param,fname,*args)
    command=f'{NASTRAN_LOC} {fname} out={self.dir_output} old=no'
    os.system(f'{command} > nul 2>&1')
    self.nid_rom,self.Ka,self.Ma=read_pch(f'{self.dir_output}rom.pch')
  
  def get_rom(self,param,*args):
    self._run_nastran(param,*args)
    self.eigvals,self.eigvecs=sp.linalg.eigh(self.Ka,self.Ma+np.eye(self.Ma.shape[0])*1e-4)
    return self.Ka,self.Ma,self.eigvals,self.eigvecs
  
  def get_aero(self,param,*args):
    #get eigenvalues and eigenvectors (full dimension)
    fname=f'{self.dir_input}rom_f.bdf'
    self.amodel.return_bdf(param,fname,*args)
    command=f'{NASTRAN_LOC} {fname} out={self.dir_output} old=no'
    os.system(f'{command} > nul 2>&1')
    op2model=read_op2(f'{self.dir_output}rom_f.op2',debug=None)
    modesdata=op2model.eigenvectors[1].data
    eigsdata=op2model.eigenvectors[1].eigns
    self.modesdata=modesdata
    self.eigsdata=eigsdata
    #write eigvecs and eigvals to op4 file
    generate_aero(modesdata,eigsdata)

    #calculate aero using Nastran
    fname=f'{self.dir_input}rom_aero.bdf'
    aemodel=self.cls_NatranModel(self.fname_aeroelastic)
    aemodel.return_bdf(param,fname,*args)
    command=f'{NASTRAN_LOC} {fname} out={self.dir_output} old=no'
    os.system(f'{command} > nul 2>&1')

    #calculate Roger's rational function approximation
    aero=read_op4('./data_out/Qhh50-50.op4')
    aero2=read_op4('./data_out/Qhj0_8-50.op4')
    qhh = jnp.array(aero['Q_HH'].data)
    qhj =  jnp.array(aero2['Q_HJ'].data)
    k_array = jnp.linspace(1e-3,1, 50)
    self.poles=jnp.linspace(1e-3,5,20)
    k_matrix_comp = frequency_matrix(k_array, self.poles)
    self.rogerQhh=rogerRFA(k_matrix_comp,qhh)
    self.rogerQhj=rogerRFA(k_matrix_comp,qhj)

    #extract eigenvector of reduced order model
    nid_aset=self.amodel.bdf_handler.bdf.asets[0].node_ids
    nid_aset=np.sort(nid_aset)
    nid_aset_in_full=_find_indices(nid_aset,op2model.eigenvectors[1].node_gridtype[:,0])
    self.eigvecs_aset=modesdata[:,nid_aset_in_full]
    self.nid_aset=nid_aset
    
    #extract coordinates of reduced order model
    nodes=self.amodel.bdf_handler.bdf.Nodes(nid_aset)
    self.coord_aset=np.array([node.xyz for node in nodes])

  def set_aeromodel(self,fname_aero,fname_aeroelastic):
    """
    fname_aero : run_caof.bdf
    fname_aeroelastic : run_cao.bdf
    """
    self.amodel=self.cls_NatranModel(fname_aero)
    self.fname_aeroelastic=fname_aeroelastic
    
class SensitivityNastran(NastranHandler):
  def __init__(self,cls_NatranModel:type(NastranDesignModel),bdfname,
               dir_input:str,dir_output:str,delta=1e-4,n_parallel=5,max_memory=0.8):
    self.cls_NatranModel=cls_NatranModel
    self.model=cls_NatranModel(bdfname)
    self.dir_input=dir_input
    self.dir_output=dir_output
    self.delta=delta
    self.n_parallel=n_parallel
    self.max_memory=max_memory

  def _run_nastran_parallel(self,param,*args):
    nparam=param.shape[0]
    fnames=[f'{self.dir_input}rom_{i}.bdf' for i in range(nparam*2)]
    params=param.repeat(nparam*2).reshape(-1,nparam*2).T
    for i in range(nparam*2):
      if i%2==0:
        params[i,i//2]-=self.delta
      else:
        params[i,i//2]+=self.delta
    self.model.return_bdfs(params,fnames,*args)
    parallel_execute_nastran(self.dir_input,self.dir_output,'rom_',nparam*2,
                             self.n_parallel,self.max_memory)
    #read Ka and Ma from pch files
    Ka_list=[]
    Ma_list=[]
    eigvec_list=[]
    eigval_list=[]
    for i in range(nparam*2):
      _,Ka,Ma=read_pch(f'{self.dir_output}rom_{i}.pch')
      eigvals,eigvecs=sp.linalg.eigh(Ka, Ma+np.eye(Ma.shape[0])*1e-4)
      Ka_list.append(Ka)
      Ma_list.append(Ma)
      eigvec_list.append(eigvecs)
      eigval_list.append(eigvals)

    # calculate sensitivity
    dKa_list=[]
    dMa_list=[]
    dval_list=[]
    dvec_list=[]
    for i in range(nparam):
      dKa=(Ka_list[2*i+1]-Ka_list[2*i])/(2*self.delta)
      dMa=(Ma_list[2*i+1]-Ma_list[2*i])/(2*self.delta)
      dval=(eigval_list[2*i+1]-eigval_list[2*i])/(2*self.delta)
      dvec=(eigvec_list[2*i+1]-eigvec_list[2*i])/(2*self.delta)
      dKa_list.append(dKa)
      dMa_list.append(dMa)
      dval_list.append(dval)
      dvec_list.append(dvec)
    self.dKa=np.array(dKa_list)
    self.dMa=np.array(dMa_list)
    self.dval=np.array(dval_list)
    self.dvec=np.array(dvec_list)

  def _run_nastran_parallel_op2(self,param,*args):
    nparam=param.shape[0]
    fnames=[f'{self.dir_input}rom_{i}.bdf' for i in range(nparam*2)]
    params=param.repeat(nparam*2).reshape(-1,nparam*2).T
    for i in range(nparam*2):
      if i%2==0:
        params[i,i//2]-=self.delta
      else:
        params[i,i//2]+=self.delta
    self.model.return_bdfs(params,fnames,*args)
    parallel_execute_nastran(self.dir_input,self.dir_output,'rom_',nparam*2,
                             self.n_parallel,self.max_memory)
    #read eigvecs and eigvals from op2 files
    eigvec_list=[]
    eigval_list=[]
    for i in range(nparam*2):
      op2model=read_op2(f'{self.dir_output}rom_{i}.op2',debug=None)
      eigvec_list.append(op2model.eigenvectors[1].data)
      eigval_list.append(np.array(op2model.eigenvectors[1].eigns))

    # calculate sensitivity
    dvec_list=[]
    dval_list=[]
    for i in range(nparam):
      dvec=(eigvec_list[2*i+1]-eigvec_list[2*i])/(2*self.delta)
      dval=(eigval_list[2*i+1]-eigval_list[2*i])/(2*self.delta)
      dvec_list.append(dvec)
      dval_list.append(dval)
    self.dvec=np.array(dvec_list)
    self.dval=np.array(dval_list)
    
  def get_rom_sensitivity(self,param,*args):
    self._run_nastran_parallel(param,*args)
    return self.dKa,self.dMa,self.dvec,self.dval

  def get_eig_sensitivity_op2(self,param,*args):
    self._run_nastran_parallel_op2(param,*args)
    return self.dvec,self.dval

def parallel_execute_nastran(input_dir,output_dir,dbf_name,n_job,n_parallel=5,max_memory=0.8):
  fname_template=input_dir+dbf_name+'{}.bdf'
  max_memory_job=max_memory/n_parallel
  cmd_list=[]
  for i in range(n_job):
    fname=fname_template.format(i)
    command=f'{NASTRAN_LOC} {fname} out={output_dir} memorymax={max_memory_job} old=no news=no'
    cmd_list.append(f'{command} > nul 2>&1')
  
  with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
    executor.map(os.system, cmd_list)

def _find_indices(arr_trg, arr_ref):
  index_map = {value: idx for idx, value in enumerate(arr_ref)}
  return np.array([index_map.get(element, None) for element in arr_trg])