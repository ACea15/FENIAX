import jax
import jax.numpy as jnp
import numpy as np

def _grad_eig(A,B,w,v,dA,dB,inv_A_lB):
  """
  Calculate the gradient of eigenvalues with respect to A and B
  A: symmetric matrix (na,na)
  B: symmetric matrix (na,na)
  w: eigenvalues (nl,)
  v: eigenvectors (nv,nl)
  dA: perturbation of A (np,na,na)
  dB: perturbation of B (np,na,na)
  w,v=sp.linalg.eigh(A,B)

  Returns:
  dv: gradient of eigenvectors (np,nv,nl)
  dw: gradient of eigenvalues (np,nl)
  """
  _v=v.T #(nl,nv)
  _dw_dA=(_v[:,:,None]*_v[:,None,:]) #(nl,na,na)
  dw_dA=(_dw_dA*dA[:,None]).sum(axis=(-1,-2)) #(np,nl)
  dw_dB=-(_dw_dA*w[:,None,None]*dB[:,None]).sum(axis=(-1,-2)) #(np,nl)
  #A_lB=A-B*w[:,None,None] #(nl,na,na)
  #inv_A_lB=np.linalg.pinv(A_lB) #(nl,nv,nv)
  dAdAv=_diag_multi(_v,dA) #(np,nl,nv)
  dBdBv=_diag_multi(_v,dB) #(np,nl,nv)
  termA=dw_dA[:,:,None]*(B@v).T-dAdAv #(np,nl,nv)
  termB=dw_dB[:,:,None]*(B@v).T+dw_dB[:,:,None]*dBdBv #(np,nl,nv)
  dv_dA=(termA[:,:,None]@inv_A_lB).reshape(termA.shape) #(np,nl,nv)
  dv_dB=(termB[:,:,None]@inv_A_lB).reshape(termA.shape) #(np,nl,nv)
  dv=(dv_dA+dv_dB).transpose(0,2,1) #(np,nv,nl)
  dw=dw_dA+dw_dB
  return dv,dw

def _diag_multi(v,d):
  """
  v: (nl,nv)
  d: (np,na,na)
  out: (np,nl,nv)
  """
  out=v[None]@d
  return out

def shift_mat(Ma):
  threashold=1e-5
  mineigval=np.linalg.eigvalsh(Ma).min()
  out=Ma
  if mineigval<=0:
    out=Ma+(threashold-mineigval)*np.eye(Ma.shape[0])
  return out

def _T(x):
  return jnp.swapaxes(x, -1, -2)

def _H(x):
  return jnp.conj(_T(x))

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
  dv = jax.vmap(
      lambda vi, wi: -wi @ db @ wi * vi + wi @ da @ wi, in_axes=(0, 1),)(v, w)
  dv = dv.T

  E = v[None,:] - v[:,None]

  # diagonal entries: compute as column then put into diagonals
  term=(-0.5 * jax.vmap(lambda wi: wi @ db @ wi, in_axes=1)(w)).T
  diags=jnp.zeros((npa,na,na)).at[:,jnp.arange(na),jnp.arange(na)].set(term)
  
  # off-diagonals: there will be NANs on the diagonal, but these aren't used
  off_diags = jnp.reciprocal(E)*(_H(w) @ (da @ w - db @ w * v[jnp.newaxis, :]))

  dw = w @ jnp.where(jnp.eye(a.shape[0], dtype=bool), diags, off_diags)

  return dv, dw