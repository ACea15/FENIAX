"""
Class name: '{nastran card name}'+'{arbitrary label}' (e.g. 'PSHELL'+'T')
"""

import scipy as sp
from composite_material import *

class BUGDecoder:
  is_variable=True #if True, the corresponding parameter is put into sensitivity analysis
  scale_param=1.0
  idx=None
  def get_val(self,param):
    raise NotImplementedError('get_val must be implemented in the derived class')

class PSHELLT(BUGDecoder):
  def __init__(self,pid,bug_handler,y_control=None):
    self.idx=pid
    self.pshell_y_abs=np.abs(bug_handler.get_pshell_coordinates(pid)[:,1])
    self.coord_control=y_control
    self.bug_handler=bug_handler
  
  def get_val(self,param):
    """
    param : thickness ratio at control points (nparam,)
    """
    nparam=len(param)
    if self.coord_control is None:
      self.coord_control=np.linspace(self.pshell_y_abs.min(),self.pshell_y_abs.max(),nparam)
    #interpolate thickness using PchipInterpolator
    thick_ratio=sp.interpolate.PchipInterpolator(self.coord_control,param)(self.pshell_y_abs)
    thickness=thick_ratio*self.bug_handler.get_pshell_thickness(self.idx)
    #return dictionary
    out=dict()
    out['pid']=self.idx
    out['t']=thickness
    return out
  
class MAT2G(BUGDecoder):
  def __init__(self,pid,bug_handler,g_lamina,y_control=None):
    mid=bug_handler.get_mid_from_pid(pid)
    assert len(mid)==np.unique(mid).shape[0], 'MAT2 must be unique in given pid set'
    self.idx=pid
    self.mid=mid
    pshell_y=bug_handler.get_pshell_coordinates(pid)[:,1]
    self.pshell_y_abs=np.abs(pshell_y)
    self.pshell_y_sgn=np.sign(pshell_y)
    self.coord_control=y_control
    self.bug_handler=bug_handler
    self.g_lamina=g_lamina #lamina properties
  
  def get_val(self,param):
    """
    param : theta and alpha at control points (nparam,2)
      theta : angle between x-axis and material principle axis in deree
      alpha : stacking angle [0,+alpha,-alpha,90]s in degree
    """
    self.scale_param=180.0
    nparam=len(param)
    if self.coord_control is None:
      self.coord_control=np.linspace(self.pshell_y_abs.min(),self.pshell_y_abs.max(),nparam)
    #interpolate angles using PchipInterpolator
    theta=sp.interpolate.PchipInterpolator(self.coord_control,param[:,0])(self.pshell_y_abs)
    theta=self.pshell_y_sgn*theta*np.pi/180
    alpha=sp.interpolate.PchipInterpolator(self.coord_control,param[:,1])(self.pshell_y_abs)
    alpha=alpha*np.pi/180
    
    #calculate Q matrix
    gmat=gmat_from_alpha_and_theta(alpha,theta,self.g_lamina)
    self.alpha=alpha
    self.theta=theta
    #return dictionary
    out=dict()
    out['mid']=self.mid
    out['G11']=gmat[:,0,0]
    out['G12']=gmat[:,0,1]
    out['G13']=gmat[:,0,2]
    out['G22']=gmat[:,1,1]
    out['G23']=gmat[:,1,2]
    out['G33']=gmat[:,2,2]
    return out
  
class CONM2X1(BUGDecoder):
  def __init__(self,eid,bug_handler,y_control=None):
    self.idx=eid
    self.bug_handler=bug_handler
    conm_y=bug_handler.get_conm_coordinates(eid)[:,1]
    self.coord_control=y_control
    self.conm_y_abs=np.abs(conm_y)
    self.coord_control=y_control
    self.bug_handler=bug_handler

  def get_val(self,param):
    """
    param : x component of mass offset at control points (nparam,)
    """
    nparam=len(param)
    if self.coord_control is None:
      self.coord_control=np.linspace(self.conm_y_abs.min(),self.conm_y_abs.max(),nparam)
    #interpolate mass using PchipInterpolator
    x=sp.interpolate.PchipInterpolator(self.coord_control,param)(self.conm_y_abs)
    offset=self.bug_handler.get_conm_offset(self.idx)
    offset[:,0]+=x
    #return dictionary
    out=dict()
    out['eid']=self.idx
    out['X']=offset
    return out
    
class CAERO1PX(BUGDecoder):
  def __init__(self,eid,bug_handler,y_control=None):
    self.bug_handler=bug_handler
    self.idx=eid
    self.coord_control=bug_handler.get_caero_coordinates_y_abs(eid)
    
  def get_val(self,param):
    """
    param : offset of x component of P1, P4 
    """
    assert len(param)==len(self.coord_control), f'param must have the same length as y_control {self.coord_control.shape}'
    #return dictionary
    out=dict()
    out['eid']=self.idx
    out['p1']=[]
    out['p4']=[]
    interpolator=sp.interpolate.PchipInterpolator(self.coord_control,param)
    for e in self.idx:
      p1=self.bug_handler.bdf.caeros[e].p1.copy()
      offsetx1=interpolator(np.abs(p1[1]))
      p1[0]+=offsetx1
      out['p1'].append(p1)
      p4=self.bug_handler.bdf.caeros[e].p4.copy()
      offsetx4=interpolator(np.abs(p4[1]))
      p4[0]+=offsetx4
      out['p4'].append(p4)
    out['eid']=np.array(out['eid'])
    out['p1']=np.array(out['p1'])
    out['p4']=np.array(out['p4'])
    return out
  
class CAERO1CHORD(CAERO1PX):
  def get_val(self,param):
    """
    param : offset of x component of P1, P4 
    """
    #return dictionary
    out=dict()
    out['eid']=self.idx
    out['x12']=[]
    out['x43']=[]
    interpolator=sp.interpolate.PchipInterpolator(self.coord_control,param)
    for e in self.idx:
      p1=self.bug_handler.bdf.caeros[e].p1.copy()
      x12=self.bug_handler.bdf.caeros[e].x12
      x12+=interpolator(np.abs(p1[1]))
      out['x12'].append(x12)
      p4=self.bug_handler.bdf.caeros[e].p4.copy()
      x43=self.bug_handler.bdf.caeros[e].x43
      x43+=interpolator(np.abs(p4[1]))
      out['x43'].append(x43)
    out['eid']=np.array(out['eid'])
    out['x12']=np.array(out['x12'])
    out['x43']=np.array(out['x43'])
    return out

