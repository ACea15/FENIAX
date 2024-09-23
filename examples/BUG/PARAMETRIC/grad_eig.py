import jax.numpy as jnp
import numpy as np

def shift_mat(Ma):
  threashold=1e-5
  mineigval=np.linalg.eigvalsh(Ma).min()
  out=Ma
  if mineigval<=0:
    out=Ma+(threashold-mineigval)*np.eye(Ma.shape[0])
  return out

def grad_eig(a, b, v, w, da, db):
  """
  a: (n,n) symmetric matrix
  b: (n,n) symmetric matrix
  v: (n,) eigenvalues
  w: (n,n) eigenvectors
  da: (np,n,n) derivatie of a
  db: (np,n,n) derivatie of b
  """
  na = a.shape[0]
  npa=da.shape[0]
  # compute only the diagonal entries
  dval=-((db@w)*w).sum(axis=-2)*v+((da@w)*w).sum(axis=-2)

  E = v[None,:]-v[:,None]+np.eye(v.shape[0])
  
  # diagonal entries: compute as column then put into diagonals
  term=-0.5*((db@w)*w).sum(axis=-2)
  diags=np.zeros((npa,na,na))
  diags[:,np.arange(na),np.arange(na)]=term
  
  # off-diagonals: there will be NANs on the diagonal, but these aren't used
  off_diags=(w.T@(da@w-db@w*v[jnp.newaxis,:]))/E
  dvec=w@np.where(np.eye(a.shape[0],dtype=bool),diags,off_diags)
  return dval, dvec