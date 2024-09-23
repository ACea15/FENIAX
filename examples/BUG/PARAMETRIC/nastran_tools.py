import concurrent.futures
import os
import numpy as np
from pyNastran.bdf.bdf import read_bdf

NASTRAN_LOC=None

def parallel_execute_nastran(input_dir,output_dir,dbf_name,n_job,n_parallel=5,max_memory=0.8):
  if NASTRAN_LOC is None:
    raise ValueError('NASTRAN_LOC is not defined')
  fname_template=input_dir+dbf_name+'{}.bdf'
  max_memory_job=max_memory/n_parallel
  cmd_list=[]
  for i in range(n_job):
    fname=fname_template.format(i)
    command=f'{NASTRAN_LOC} {fname} out={output_dir} memorymax={max_memory_job} old=no news=no'
    cmd_list.append(f'{command} > nul 2>&1')
  
  with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
    executor.map(os.system, cmd_list)

def read_pch(fname):
  pchmodel=read_bdf(fname,punch=True,debug=None)
  Kaa=pchmodel.dmig['KAAX'].get_matrix()[0]
  Maa=pchmodel.dmig['MAAX'].get_matrix()[0]
  idx=pchmodel.dmig['KAAX'].get_matrix()[1]
  nid_rom=[]
  for i in idx:
    if idx[i][1]==1:
      nid_rom.append(idx[i][0])
  nid_rom=np.array(nid_rom)
  Kaa=Kaa.astype(np.float64)
  Maa=Maa.astype(np.float64)
  return Kaa,Maa,nid_rom