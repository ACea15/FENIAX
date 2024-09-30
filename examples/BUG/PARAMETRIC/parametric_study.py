import itertools
from scipy.stats import qmc

def decode_params(params,doe=False,nsample=10)->list[dict]:
  """
  If doe is False, it will return a full grid of the parameters
  If doe is True, it will return a Latin Hypercube of the parameters,
  in which only the first entries of each list will be used as the lower bound
  and the last as the upper bound.


  params['key']=[[val1_1,val1_2,...],[val2_1,val2_2,...],...]

  temp=[[[val1_1,val2_1,...],[val1_2,val2_1,...],...],
        [[val1_1,val2_2,...],[val1_2,val2_2,...],...],
        ...]

  out=[{'key':val1_1,'key2':val2_1,...},
        {'key':val1_2,'key2':val2_1,...},
        ...]
  """
  out=[]
  if not doe: # full grid
    temp=[]
    for key in params:
      vals=params[key]
      temp.append(list(itertools.product(*vals)))
    out_vals=list(itertools.product(*temp))
    for vals in out_vals:
      out.append({key:list(val) for key,val in zip(params.keys(),vals)})
  elif doe: # design of experiments (Latin Hypercube)
    ndim=0
    lowers=[]
    uppers=[]
    sids=[]
    eids=[]
    sid=0
    for key in params:
      vals=params[key]
      ndim+=len(vals)
      eid=sid+len(vals)
      sids.append(sid)
      eids.append(eid)
      sid=eid
      for val in vals:
        lowers.append(val[0])
        uppers.append(val[-1])
    sampler=qmc.LatinHypercube(d=ndim,optimization="random-cd")
    sample=sampler.random(n=nsample)
    sample_scaled=qmc.scale(sample,lowers,uppers)
    for i in range(nsample):
      out.append({key:list(sample_scaled[i][sid:eid]) for key,sid,eid in zip(params.keys(),sids,eids)})
  return out


