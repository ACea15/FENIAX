import numpy as np
import scipy as sp

def read_pch(fname):
  f=open(fname,'r')
  lines=f.read().splitlines()
  f.close()
  indice_k=[]
  data_k=[]
  indice_m=[]
  data_m=[]
  id1=0
  nid_list=[]
  for line in lines:
    if 'DMIG*   KAAX' in line:
      _line=line.split()
      id1=int(_line[2])*6+int(_line[3])-1
      if _line[3]=='1':
        nid_list.append(int(_line[2]))
      Flag_k=True
    elif 'DMIG*   MAAX' in line:
      _line=line.split()
      id1=int(_line[2])*6+int(_line[3])-1
      Flag_k=False
    elif '*' in line:
      _line=line.split()
      if len(_line)==3: #negative value
        dim=int(_line[-1][0])
        val=float(_line[-1][1:])
      elif len(_line)==4: #positive value
        dim=int(_line[-2])
        val=float(_line[-1])
      else: raise ValueError('Unknown format')
      id2=int(_line[1])*6+dim-1
      if Flag_k:
        indice_k.append([id1,id2])
        data_k.append(val)
      else:
        indice_m.append([id1,id2])
        data_m.append(val)
  data_k=np.array(data_k)
  data_m=np.array(data_m)
  dim_original=np.repeat(nid_list,6)*6+np.repeat(np.arange(6),len(nid_list)).reshape(6,-1).T.flatten()
  dim_condensed=np.arange(len(nid_list)*6)
  dic_dim=dict(zip(dim_original,dim_condensed))
  # replace indices
  indice_k_=np.array([dic_dim[i] for i in np.array(indice_k).flatten()]).reshape(-1,2)
  indice_m_=np.array([dic_dim[i] for i in np.array(indice_m).flatten()]).reshape(-1,2)
  mat_shape=(len(nid_list)*6,len(nid_list)*6)
  # make symmetric
  msk_nondiag_k=(indice_k_[:,0]!=indice_k_[:,1])
  indice_k=np.concatenate([indice_k_,indice_k_[msk_nondiag_k,::-1]],axis=0)
  data_k=np.concatenate([data_k,data_k[msk_nondiag_k]],axis=0)
  msk_nondiag_m=(indice_m_[:,0]!=indice_m_[:,1])
  indice_m=np.concatenate([indice_m_,indice_m_[msk_nondiag_m,::-1]],axis=0)
  data_m=np.concatenate([data_m,data_m[msk_nondiag_m]],axis=0)

  Kaa=sp.sparse.coo_matrix((data_k,(indice_k[:,0],indice_k[:,1])),shape=mat_shape).toarray()
  Maa=sp.sparse.coo_matrix((data_m,(indice_m[:,0],indice_m[:,1])),shape=mat_shape).toarray()
  return nid_list,Kaa,Maa