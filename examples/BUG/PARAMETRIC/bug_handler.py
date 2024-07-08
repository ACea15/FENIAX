#Class for preprocessing BUG model

import concurrent.futures
from pyNastran.bdf.bdf import read_bdf
from pyNastran.op2.op2 import OP2
import plotly.graph_objects as go
import numpy as np
import os

NASTRAN_LOC='cmd.exe /c C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe'


class BUGHandler:
  def __init__(self,fname):
    self.bdf_name=fname
    self.bdf = read_bdf(fname,debug=None)
    self._preprocess_properties()
    self._preprocess_nodes()
    self._preprocess_elements()
    self._preprocess_materials()

  def _preprocess_properties(self):
    """
    split pids into types
    """
    pid_dict={}
    ptype=[]
    for pid,propcard in self.bdf.properties.items():
      if propcard.type not in ptype:
        ptype.append(propcard.type)
        pid_dict[propcard.type]=[pid]
      else:
        pid_dict[propcard.type].append(pid)
    self.pid_dict=pid_dict
    self.ptypes=ptype

  def _preprocess_nodes(self):
    """
    get all nodes
    """
    self.nnode=len(self.bdf.nodes)
    self.nids=list(self.bdf.nodes.keys())
    node_coords=np.zeros((self.nnode,3))
    for i in range(self.nnode):
      node_coords[i]=self.bdf.nodes[self.nids[i]].xyz
    self.node_coords=node_coords  

    nodes=np.zeros((self.nnode,3))
    for i,nid in enumerate(np.sort(self.nids)):
      nodes[i]=self.bdf.nodes[nid].xyz
    self.nodes=nodes

  def _preprocess_elements(self):
    """
    split eids into types
    """
    eid_dict={}
    etype=[]
    for eid,elem in self.bdf.elements.items():
      if elem.type not in etype:
        etype.append(elem.type)
        eid_dict[elem.type]=[eid]
      else:
        eid_dict[elem.type].append(eid)
    self.eid_dict=eid_dict
    self.etypes=etype

  def _preprocess_materials(self):
    """
    split mids into types
    """
    mid_dict={}
    mtype=[]
    for mid,mat in self.bdf.materials.items():
      if mat.type not in mtype:
        mtype.append(mat.type)
        mid_dict[mat.type]=[mid]
      else:
        mid_dict[mat.type].append(mid)
    self.mid_dict=mid_dict
    self.mtypes=mtype

  def get_nodes_using_pids(self,pids):
    """
    get nids in elements that use given property ids
    """
    nid_trg=[]
    eid_trg=[]
    for eid,elem in self.bdf.elements.items():
      if elem.pid in pids:
        nid_trg+=elem.nodes
        eid_trg.append(eid)
    nid_trg=list(set(nid_trg))
    eid_trg=list(set(eid_trg))
    coord=np.zeros((len(nid_trg),3))
    for i,nid in enumerate(nid_trg):
      coord[i]=self.bdf.nodes[nid].xyz
    return nid_trg,coord,eid_trg
  
  def get_pids_from_nids(self,nids):
    """
    get pids of elements that use given node ids
    """
    pids=[]
    for eid,elem in self.bdf.elements.items():
      for nid in elem.nodes:
        if nid in nids:
          pids.append(elem.pid)
          break
    return list(set(pids))
  
  def plot_nodes(self,marker_size=3):
    """
    plot nodes
    """
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=self.node_coords[:,0],y=self.node_coords[:,1],z=self.node_coords[:,2],
                               mode='markers',marker=dict(size=marker_size),hovertext=self.nids,name='All nodes'))
    fig.update_layout(scene=dict(aspectmode='data'),margin=dict(l=0,r=0,b=0,t=10))
    return fig
  
  def plot_elems(self,fig=None,msk_plot=[2,3,4],show_mesh=False):
    """
    plot elements
    """
    if fig==None:
      fig=go.Figure()

    hovertemplate='<b>PID:</b> %{text}<br><b>MID:</b> %{customdata}<br><extra></extra>'
    for etype in self.etypes:
      eid=self.eid_dict[etype]
      elems=self.bdf.Elements(eid)
      nelem=len(elems)
      nnode=len(elems[0].nodes)
      pids=np.zeros(nelem)
      mids=np.zeros(nelem)
      if nnode==2 and (nnode in msk_plot):
        coord=np.zeros((nelem,2,3)) #(ne,2,3)
        for i,elem in enumerate(elems):
          for j,nid in enumerate(elem.nodes):
            coord[i,j]=self.bdf.nodes[nid].xyz
          pids[i]=elem.pid
          mids[i]=self.bdf.properties[elem.pid].mid
        pname=self.bdf.properties[elem.pid].type
        nones=np.array([None]*nelem*3).reshape(nelem,1,3)
        edges=np.concatenate((coord,nones),axis=1).reshape(-1,3) #(ne*3,3)
        fig.add_trace(go.Scatter3d(x=edges[:,0],y=edges[:,1],z=edges[:,2],mode='lines',name=etype,hoverinfo='skip'))
        center=coord.mean(axis=1) #(ne,3)
        
        fig.add_trace(go.Scatter3d(x=center[:,0],y=center[:,1],z=center[:,2],mode='markers',marker=dict(size=2),customdata=mids,
                                    text=pids,hovertemplate=hovertemplate,name=pname))
      elif nnode==3 and (nnode in msk_plot):
        nelem=len(elems)
        coord=np.zeros((nelem,3,3))
        for i,elem in enumerate(elems):
          for j,nid in enumerate(elem.nodes):
            coord[i,j]=self.bdf.nodes[nid].xyz
          pids[i]=elem.pid
          mids[i]=self.bdf.properties[elem.pid].mid1
        pname=self.bdf.properties[elem.pid].type
        
        idx=np.arange(nelem*3).reshape(-1,3).T #(3,ne) 
        center=coord.mean(axis=1)
        coord=coord.reshape(-1,3) #(ne*3,3)
        if show_mesh:
          fig.add_trace(go.Mesh3d(x=coord[:,0],y=coord[:,1],z=coord[:,2],i=idx[0],j=idx[1],k=idx[2],name=etype,opacity=0.3,hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=center[:,0],y=center[:,1],z=center[:,2],mode='markers',marker=dict(size=2),customdata=mids,
                                  text=pids,hovertemplate=hovertemplate,name=pname))
      elif nnode==4 and (nnode in msk_plot):
        nelem=len(elems)
        coord=np.zeros((nelem,4,3))
        for i,elem in enumerate(elems):
          for j,nid in enumerate(elem.nodes):
            coord[i,j]=self.bdf.nodes[nid].xyz
          pids[i]=elem.pid
          mids[i]=self.bdf.properties[elem.pid].mid1
        pname=self.bdf.properties[elem.pid].type
        idx=np.arange(nelem*4).reshape(-1,4)[:,[0,1,2,0,2,3]].reshape(-1,3).T #(3,ne*2)
        center=coord.mean(axis=1)
        coord=coord.reshape(-1,3)
        if show_mesh:
          fig.add_trace(go.Mesh3d(x=coord[:,0],y=coord[:,1],z=coord[:,2],i=idx[0],j=idx[1],k=idx[2],name=etype,opacity=0.3,hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=center[:,0],y=center[:,1],z=center[:,2],mode='markers',marker=dict(size=2),customdata=mids,
                                  text=pids,hovertemplate=hovertemplate,name=pname))
    fig.update_layout(scene=dict(aspectmode='data'),margin=dict(l=0,r=0,b=0,t=10))
    return fig
    
  def plot_shells(self,pids,etypes=['CTRIA3', 'CQUAD4'],fig=None,showlegend=False):
    """
    plot shell elements that have pshells of given pids
    """
    if fig is None:
      fig=go.Figure()
    coord_list=[]
    connect_list=[]
    connect_offset=0
    pid_tri=[]
    pid_quad=[]
    label_tri=[]
    label_quad=[]
    for etype in etypes:
      eid_trg_dict={}
      for pid in pids:
        eid_trg_dict[pid]=[]
      elems=self.bdf.Elements(self.eid_dict[etype])
      nnode=len(elems[0].nodes)
      for elem in elems:
        if elem.pid in pids:
          eid_trg_dict[elem.pid].append(elem.eid)
      if nnode==3:
        for pid in pids:
          
          eid=eid_trg_dict[pid]
          nelem=len(eid)
          if nelem==0:
            continue
          elems=self.bdf.Elements(eid)
          coord=np.zeros((nelem,3,3))
          for i,elem in enumerate(elems):
            pid_tri.append(elem.pid)
            for j,nid in enumerate(elem.nodes):
              coord[i,j]=self.bdf.nodes[nid].xyz
            norm=np.cross(coord[i,1]-coord[i,0],coord[i,2]-coord[i,0])
            label_tri.append(norm[2]>0)
          idx=np.arange(nelem*3).reshape(-1,3).T #(3,ne) 
          coord=coord.reshape(-1,3) #(ne*3,3)
          if showlegend:
            fig.add_trace(go.Mesh3d(x=coord[:,0],y=coord[:,1],z=coord[:,2],i=idx[0],j=idx[1],k=idx[2],name=f'PID: {pid} ({etype})',opacity=0.3,showlegend=showlegend))
          else:
            coord_list.append(coord)
            connect_list.append(idx+connect_offset)
            connect_offset+=len(coord)
      elif nnode==4:
        for pid in pids:
          eid=eid_trg_dict[pid]
          nelem=len(eid)
          if nelem==0:
            continue
          elems=self.bdf.Elements(eid)
          coord=np.zeros((nelem,4,3))
          for i,elem in enumerate(elems):
            pid_quad.append(elem.pid)
            for j,nid in enumerate(elem.nodes):
              coord[i,j]=self.bdf.nodes[nid].xyz
            norm=np.cross(coord[i,1]-coord[i,0],coord[i,2]-coord[i,0])
          label_quad.append(norm[2]>0)
          idx=np.arange(nelem*4).reshape(-1,4)[:,[0,1,2,0,2,3]].reshape(-1,3).T #(3,ne*2)
          coord=coord.reshape(-1,3) #(ne*3,3)
          if showlegend:
            fig.add_trace(go.Mesh3d(x=coord[:,0],y=coord[:,1],z=coord[:,2],i=idx[0],j=idx[1],k=idx[2],name=f'PID: {pid} ({etype})',opacity=0.3,showlegend=showlegend))
          else:
            coord_list.append(coord)
            connect_list.append(idx+connect_offset)
            connect_offset+=len(coord)
    if not showlegend:
      coord=np.concatenate(coord_list,axis=0)
      idx=np.concatenate(connect_list,axis=1)
      fig.add_trace(go.Mesh3d(x=coord[:,0],y=coord[:,1],z=coord[:,2],i=idx[0],j=idx[1],k=idx[2],opacity=0.3))
    fig.update_layout(scene=dict(aspectmode='data'),margin=dict(l=0,r=0,b=0,t=10))
    return fig

  def get_pshell_info(self,pids,etypes=['CTRIA3', 'CQUAD4']):
    """
    get pshell information
    """
    coord_list=[]
    connect_list=[]
    connect_offset=0
    pid_tri=[]
    pid_quad=[]
    label_tri=[]
    label_quad=[]
    for etype in etypes:
      eid_trg_dict={}
      for pid in pids:
        eid_trg_dict[pid]=[]
      elems=self.bdf.Elements(self.eid_dict[etype])
      nnode=len(elems[0].nodes)
      for elem in elems:
        if elem.pid in pids:
          eid_trg_dict[elem.pid].append(elem.eid)
      if nnode==3:
        for pid in pids:
          
          eid=eid_trg_dict[pid]
          nelem=len(eid)
          if nelem==0:
            continue
          elems=self.bdf.Elements(eid)
          coord=np.zeros((nelem,3,3))
          for i,elem in enumerate(elems):
            pid_tri.append(elem.pid)
            for j,nid in enumerate(elem.nodes):
              coord[i,j]=self.bdf.nodes[nid].xyz
            norm=np.cross(coord[i,1]-coord[i,0],coord[i,2]-coord[i,0])
            label_tri.append(norm[2]>0)
          idx=np.arange(nelem*3).reshape(-1,3).T #(3,ne) 
          coord=coord.reshape(-1,3) #(ne*3,3)
          coord_list.append(coord)
          connect_list.append(idx+connect_offset)
          connect_offset+=len(coord)
      elif nnode==4:
        for pid in pids:
          eid=eid_trg_dict[pid]
          nelem=len(eid)
          if nelem==0:
            continue
          elems=self.bdf.Elements(eid)
          coord=np.zeros((nelem,4,3))
          for i,elem in enumerate(elems):
            pid_quad.append(elem.pid)
            for j,nid in enumerate(elem.nodes):
              coord[i,j]=self.bdf.nodes[nid].xyz
            norm=np.cross(coord[i,1]-coord[i,0],coord[i,2]-coord[i,0])
          label_quad.append(norm[2]>0)
          idx=np.arange(nelem*4).reshape(-1,4).T #(4,ne)
          coord=coord.reshape(-1,3) #(ne*3,3)
          coord_list.append(coord)
          connect_list.append(idx+connect_offset)
          connect_offset+=len(coord)
    return coord,idx,(pid_tri,pid_quad),(label_tri,label_quad)

  def set_trg_pids(self,pids):
    """
    set target pids for pshell elements to be design variables
    return initial values
    """
    self.trg_pids=pids
    n_trg_pid=len(pids)
    properties=self.bdf.Properties(pids)
    initial_thickness=np.zeros(n_trg_pid)
    for i,prop in enumerate(properties):
      initial_thickness[i]=prop.t
    self.initial_thickness=initial_thickness
    return initial_thickness
  
  def set_trg_mids(self,pids):
    """
    set target mids for pshell elements to be design variables
    return initial values
    """
    self.trg_mids=self.get_mid_from_pid(pids)
    self.center_coord_mids=self.get_pshell_coordinates(pids)
    materials=self.bdf.Materials(self.trg_mids)
    g11=np.zeros(len(materials))
    g12=np.zeros(len(materials))
    g22=np.zeros(len(materials))
    g33=np.zeros(len(materials))
    for i,mat in enumerate(materials):
      g11[i]=mat.G11
      g12[i]=mat.G12
      g22[i]=mat.G22
      g33[i]=mat.G33
    gmat_ref=np.zeros((len(materials),3,3))
    gmat_ref[:,0,0]=g11
    gmat_ref[:,0,1]=g12; gmat_ref[:,1,0]=g12
    gmat_ref[:,1,1]=g22
    gmat_ref[:,2,2]=g33
    self.gmat_ref=gmat_ref

    return self.trg_mids,self.center_coord_mids

  def set_pshell_thickness(self,thickness):
    assert len(thickness)==len(self.initial_thickness)
    for i,pid in enumerate(self.trg_pids):
      self.bdf.Properties([pid])[0].t=thickness[i]

  def set_shell_materials(self,gmat):
    assert len(gmat)==len(self.trg_mids)
    g11=gmat[:,0,0]; g12=gmat[:,0,1]; g13=gmat[:,0,2]
    g22=gmat[:,1,1]; g23=gmat[:,1,2]; g33=gmat[:,2,2]
    #for i,mat in enumerate(materials):
    for i,mid in enumerate(self.trg_mids):
      self.bdf.Materials([mid])[0].G11=g11[i]
      self.bdf.Materials([mid])[0].G12=g12[i]
      self.bdf.Materials([mid])[0].G13=g13[i]
      self.bdf.Materials([mid])[0].G22=g22[i]
      self.bdf.Materials([mid])[0].G23=g23[i]
      self.bdf.Materials([mid])[0].G33=g33[i]

  def get_mid_from_pid(self,pid):
    out=[prop.mid1 for prop in self.bdf.Properties(pid)]
    return out
  
  def get_pshell_coordinates(self,pids=None):
    if pids is None:
      pids=self.pid_dict['PSHELL']
    coords_center=np.zeros((len(pids),3))
    if self.is_eid_equal_pid():
      for i,eid in enumerate(pids):
        nids=self.bdf.elements[eid].nodes
        if len(nids)==3:
          coord=np.zeros((3,3))
        elif len(nids)==4:
          coord=np.zeros((4,3))
        for j,nid in enumerate(nids):
          coord[j]=self.bdf.nodes[nid].xyz
        coords_center[i]=coord.mean(axis=0)
      self.shell_center_coord=coords_center
      return coords_center
    else:
      raise NotImplementedError
  
  def get_mat_info_variable(self):
    mids=[]
    pids=[]
    for pid in self.pid_dict['PSHELL']:
      mids.append(self.bdf.Properties([pid])[0].mid1)
      pids.append(pid)
    _,idx,counts=np.unique(mids,return_index=True,return_counts=True)
    mids_unique=np.array(mids)[idx[counts==1]]
    pids_unique=np.array(pids)[idx[counts==1]]
    coords_center=np.zeros((len(mids_unique),3))
    label=np.zeros(len(mids_unique))
    if self.is_eid_equal_pid():
      for i,eid in enumerate(pids_unique):
        nids=self.bdf.elements[eid].nodes
        if len(nids)==3:
          coord=np.zeros((3,3))
        elif len(nids)==4:
          coord=np.zeros((4,3))
        for j,nid in enumerate(nids):
          coord[j]=self.bdf.nodes[nid].xyz
        coords_center[i]=coord.mean(axis=0)
        norm=np.cross(coord[1]-coord[0],coord[2]-coord[0])
        label[i]=norm[2]>0
      coef=np.array([coords_center[:,1],label]).T
      coef=coef/coef.max(axis=0)
      return mids_unique,pids_unique,coef
    else:
      raise NotImplementedError
  
  def is_eid_equal_pid(self):
    for eid in self.eid_dict['CQUAD4']:
      if eid-self.bdf.elements[eid].pid!=0:
        print(eid,self.bdf.elements[eid].pid)
        return False
    for eid in self.eid_dict['CTRIA3']:
      if eid-self.bdf.elements[eid].pid!=0:
        print(eid,self.bdf.elements[eid].pid)
        return False
    return True

  def run_nastran(self,thickness,out_dir):
    self.set_pshell_thickness(thickness)
    self.bdf.write_bdf('temp.bdf')
    command=f'C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe temp.bdf out={out_dir}'
    os.system(f'cmd.exe /c {command} > nul 2>&1')
    op2=OP2(debug=False)
    op2.read_op2(f'{out_dir}/temp.op2')    
    return op2
  
  def execute_nastran_multi_thickness(self,thickness_m,input_dir,out_dir):
    """
    thickness : (n,nt) where n is the number of thickness sets and nt is the number of target pids
    """
    for i,thickness in enumerate(thickness_m):
      self.set_pshell_thickness(thickness)
      fname=input_dir+f'case{i}.bdf'
      self.bdf.write_bdf(fname)
      command=f'C:/MSC.Software/MSC_Nastran/20182/bin/nast20182.exe {fname} out={out_dir}'
      os.system(f'cmd.exe /c {command} > nul 2>&1')

  def set_condition_thickness(self,pids,min_thick_ratio,max_thick_ratio):
    self.initial_thickness=self.set_trg_pids(pids)
    self.min_thick_ratio=min_thick_ratio
    self.max_thick_ratio=max_thick_ratio

  def set_condition_material(self,qmat_ref,coefs,mids,alpha_min,alpha_max,theta_min,theta_max):
    self.qmat_ref=qmat_ref
    self.coefs=coefs
    self.trg_mids=mids
    self.alpha_min=alpha_min
    self.alpha_max=alpha_max
    self.theta_min=theta_min
    self.theta_max=theta_max

def write_shell_material(gmat,mids,fname,bdf_name):
  bdf=read_bdf(bdf_name,debug=None,validate=False)
  g11=gmat[:,0,0]; g12=gmat[:,0,1]; g13=gmat[:,0,2]
  g22=gmat[:,1,1]; g23=gmat[:,1,2]; g33=gmat[:,2,2]
  for i,mid in enumerate(mids):
    bdf.Materials([mid])[0].G11=g11[i]
    bdf.Materials([mid])[0].G12=g12[i]
    bdf.Materials([mid])[0].G13=g13[i]
    bdf.Materials([mid])[0].G22=g22[i]
    bdf.Materials([mid])[0].G23=g23[i]
    bdf.Materials([mid])[0].G33=g33[i]
  bdf.write_bdf(fname)

    