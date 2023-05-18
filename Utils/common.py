import numpy as np
import pdb
import itertools
import functools
import copy
def diff_matrix(qi,q):

    dimension = np.shape(q)
    xdimension,ydimension = dimension
    qdiff = np.zeros(dimension)
    for xi in range(xdimension):
        for yi in range(ydimension):

                qdiff[xi][yi] = (q[xi][yi]-qi[xi][yi])/qi[xi][yi]

    return qdiff

def err_matrix(diff_q):

    dimension = np.shape(diff_q)
    xdimension,ydimension = dimension
    ndimension = xdimension*ydimension
    qrow = np.reshape(diff_q,(1,ndimension))
    err = np.sqrt(np.sum(qrow**2))/ndimension
    return err

def diff_tensor(qi,q):

    dimension = np.shape(q)
    dimensioni = np.shape(qi)
    assert dimension == dimensioni, 'Dimensions on inputs to diff_tensor not equal'
    qdiff = np.zeros(dimension)
    qdiff_norm = np.zeros(dimension)
    qmax = np.max(np.abs(qi))
    
    for i in itertools.product(*[range(i) for i in dimension]):
        qdiff[i] = q[i]-qi[i]
        if np.abs(qmax)>1e-7:
          qdiff_norm[i] = (q[i]-qi[i])/qmax
        else:
          #print 'qmax<1e-7 = %s' %qmax  
          qdiff_norm[i] = 0.  
    return qdiff,qdiff_norm

def err_tensor(dq,err_type='quadratic'):
    dimension = np.shape(dq)
    ndimension = functools.reduce(lambda x, y: x*y,dimension)
    qrow = np.reshape(dq,(1,ndimension))
    if err_type == 'quadratic':
        err = np.sqrt(np.sum(qrow**2))/ndimension
    elif err_type == 'linear':
        err = np.sum(np.abs(qrow))/ndimension
    return err


def flatten(lis):
    l=list(lis)
    while type(max(l)) is list and len(max(l))>0:

       for i in range(len(l)):
         if type(l[i]) is tuple:
          l[i] = list(l[i])
         if type(l[i]) is list:
          if len(l[i])==0:
           continue
          for j in range(len(l[i])):
            l.insert(i+j,l[i+j][j])
          del l[i+j+1]
    return l

def flatten2(lis):
    l=list(lis)
    i=0
#while type(max(l)) is list:

    while i < len(l):
         if type(l[i]) is tuple:
          l[i] = list(l[i])
         if type(l[i]) is list:
          if len(l[i])==0:
           del l[i]
           continue
          for j in range(len(l[i])):
            l.insert(i+j,l[i+j][j])

          del l[i+j+1]
         else:
          i=i+1

    return l

def remove_zeros(Mat,retd=None,rtol=1e-05,atol=1e-08):
    M = copy.copy(Mat)
    dofx = np.shape(M)[0]
    dofy = np.shape(M)[1]
    count=0
    dx=[]
    dy=[]
    for i in range(dofx):
        if np.allclose(Mat[i,:],np.zeros(dofy),rtol=rtol,atol=atol):
             M = np.delete(M,i-count,0)
             count+=1
             dx.append(i)
    count=0
    for i in range(dofy):
        #print M[:,i]
        #print np.zeros(dofx)
        if np.allclose(Mat[:,i],np.zeros(dofx),rtol=rtol,atol=atol):
             M =np.delete(M,i-count,1)
             count+=1
             dy.append(i)
    if retd:
        return M,dx,dy
    else:
        return M

def remove_zeros2(Mat,Mat2,retd=None,rtol=1e-05,atol=1e-08):
    M = copy.copy(Mat)
    M2 = copy.copy(Mat2)
    dofx = np.shape(M)[0]
    dofy = np.shape(M)[1]
    count=0
    dx=[]
    dy=[]
    for i in range(dofx):
        if np.allclose(Mat[i,:],np.zeros(dofy),rtol=rtol,atol=atol):
             M = np.delete(M,i-count,0)
             count+=1
             dx.append(i)
    count=0
    for i in range(dofy):
        #print M[:,i]
        #print np.zeros(dofx)
        if np.allclose(Mat[:,i],np.zeros(dofx),rtol=rtol,atol=atol):
             M =np.delete(M,i-count,1)
             count+=1
             dy.append(i)
    for i in range(dofx):
        if i in dx:
             M2 = np.delete(M2,i-count,0)
             count+=1
    count=0
    for i in range(dofy):
    
        if i in dy:
             M2 =np.delete(M2,i-count,1)
             count+=1

    if retd:
        return M,M2,dx,dy
    else:
        return M,M2
    
def remove_dot(n):
    n = str(n)
    nnew=''
    for i in n:
      if i != '.':
         nnew += i
    return nnew

def class2dic(cl):

    dic={}
    cl_attributes=dir(cl)
    for i in cl_attributes:
      dic[i] = eval('cl.%s'%i)
    return dic
      
def inverse_conn(c1):

    beams = len(c1)
    dic={}
    for bi in range(beams):
        dic[bi]=[]
    for bi in range(beams):
 
        for bj in c1[bi]:
            cx=c1[bi]+[bi]
            cx.remove(bj)
            dic[bj]=cx
    c2=[]
    for bi in range(beams):
      c2.append(dic[bi])
    return c2

        


    
