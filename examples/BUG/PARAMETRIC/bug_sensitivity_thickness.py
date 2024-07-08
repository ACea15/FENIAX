from feniax_sensitivity import NastranDesignModel
from bug_handler import BUGHandler#,write_shell_thickness
from scipy.interpolate import CloughTocher2DInterpolator
import concurrent.futures
import os
import numpy as np
from pyNastran.bdf.bdf import read_bdf

class NastranDesignModel_thickness(NastranDesignModel):
  def __init__(self,bdf_name):
    self.bdf_handler=BUGHandler(bdf_name)
    self.bdf_name=bdf_name

  def return_bdf(self,param,fname,coord,min_thick_ratio,max_thick_ratio):
    """
    param : float (nparam,)
      values of the design parameters at control points
    fname : output Nastran model
    coord : float (nparam,2)
      coordinates of the control points
    """
    assert param.max()<=1.0 and param.min()>=0.0
    interpolator=CloughTocher2DInterpolator(coord,param)
    self.bdf_handler.set_condition_thickness(self.bdf_handler.pid_dict['PSHELL'],
                                             min_thick_ratio,max_thick_ratio)
    pcoord=self.bdf_handler.get_pshell_coordinates(self.bdf_handler.trg_pids)
    pcoord=pcoord[:,:2]
    pcoord[:,1]=np.abs(pcoord[:,1])
    params=interpolator(pcoord)
    thick_ratio=min_thick_ratio+(max_thick_ratio-min_thick_ratio)*params
    thickness=self.bdf_handler.initial_thickness*thick_ratio
    write_shell_thickness(thickness,self.bdf_handler.trg_pids,fname,self.bdf_name)

  def return_bdfs(self,params,fnames,coord,min_thick_ratio,max_thick_ratio):
    self.bdf_handler.set_condition_thickness(self.bdf_handler.pid_dict['PSHELL'],
                                             min_thick_ratio,max_thick_ratio)
    pcoord=self.bdf_handler.get_pshell_coordinates(self.bdf_handler.trg_pids)
    pcoord=pcoord[:,:2]
    pcoord[:,1]=np.abs(pcoord[:,1])
    thickness_list=[]
    for param in params:
      interpolator=CloughTocher2DInterpolator(coord,param)
      _params=interpolator(pcoord)
      thick_ratio=min_thick_ratio+(max_thick_ratio-min_thick_ratio)*_params
      thickness=self.bdf_handler.initial_thickness*thick_ratio
      thickness_list.append(thickness)
    self.thickness_list=thickness_list
    self.fnames=fnames

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()-2) as executor:
      futures = []
      for i in range(len(thickness_list)):
        future=executor.submit(write_shell_thickness,thickness_list[i],
                               self.bdf_handler.trg_pids,fnames[i],self.bdf_name)
        futures.append(future)
      for future in concurrent.futures.as_completed(futures):
        pass

def write_shell_thickness(thickness,pids,fname,bdf_name):
  bdf=read_bdf(bdf_name,debug=None,validate=False)
  for i,pid in enumerate(pids):
      bdf.Properties([pid])[0].t=thickness[i]
  bdf.write_bdf(fname)