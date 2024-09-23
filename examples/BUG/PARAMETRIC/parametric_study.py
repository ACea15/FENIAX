import itertools

def decode_params(params)->list[dict]:
  """
  params['key']=[[val1_1,val1_2,...],[val2_1,val2_2,...],...]

  temp=[[[val1_1,val2_1,...],[val1_2,val2_1,...],...],
        [[val1_1,val2_2,...],[val1_2,val2_2,...],...],
        ...]

  out=[{'key':val1_1,'key2':val2_1,...},
        {'key':val1_2,'key2':val2_1,...},
        ...]
  """
  out=[]
  temp=[]
  for key in params:
    vals=params[key]
    temp.append(list(itertools.product(*vals)))
  out_vals=list(itertools.product(*temp))
  for vals in out_vals:
    out.append({key:list(val) for key,val in zip(params.keys(),vals)})
  return out


