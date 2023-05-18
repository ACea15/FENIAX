def dq12(t,q,args):


  gamma1=args[0]
  gamma2=args[1]
  eta=args[2]
  Omega=args[3]
  q1 = q[0:V.NumModes].copy()
  q2 = q[V.NumModes:2*V.NumModes].copy()
  dq1 = np.zeros(V.NumModes)
  dq2 = np.zeros(V.NumModes)
  for j in range(V.NumModes):

    for k in range(V.NumModes):
     for l in range(V.NumModes):
       dq1[j]=dq1[j]-gamma1[j,k,l]*q1[k]*q1[l]-gamma2[j,k,l]*q2[k]*q2[l]
       dq2[j]=dq2[j]+gamma2[k,j,l]*q1[k]*q2[l]

    dq1[j]=dq1[j]+Omega[j]*q2[j]+eta[j]
    dq2[j]=dq2[j]-Omega[j]*q1[j]

  return (np.hstack((dq1,dq2)))


def dq12_res(t,q,args1):

  Omega=args1['Omega']
  gamma1=args1['gamma1']
  gamma2=args1['gamma2']
  eta=args1['eta']
  NumModes=args1['NumModes']
  NumModes_res=args1['NumModes_res']
  qres=args1['qres']
  q1 = q[0:NumModes]
  q2 = q[NumModes:2*NumModes]
  q1res = qres[0:NumModes_res]
  q2res = qres[NumModes_res:2*NumModes_res]
  dq1 = np.zeros(NumModes)
  dq2 = np.zeros(NumModes)
  for j in range(NumModes):

    for k in range(NumModes):
     for l in range(NumModes):
       dq1[j]=dq1[j]-gamma1[j,k,l]*q1[k]*q1[l]-gamma2[j,k,l]*q2[k]*q2[l]
       dq2[j]=dq2[j]+gamma2[k,j,l]*q1[k]*q2[l]

    for kres in range(NumModes_res):
     for lres in range(NumModes_res):
       dq1[j]=dq1[j]-gamma1[j,kres+NumModes,lres+NumModes]*q1res[kres]*q1res[lres]-gamma2[j,kres+NumModes,lres+NumModes]*q2res[kres]*q2res[lres]
       dq2[j]=dq2[j]+gamma2[kres+NumModes,j,lres+NumModes]*q1res[kres+NumModes]*q2res[lres]

    dq1[j]=dq1[j]+Omega[j]*q2[j]+eta[j]
    dq2[j]=dq2[j]-Omega[j]*q1[j]

  return (np.hstack((dq1,dq2)))




def dJq12(t,q,args1):
  #print()
  Omega=args1[0]
  gamma1=args1[1]
  gamma2=args1[2]
  eta=args1[3]
  NumModes=len(Omega)
  q1 = q[0:NumModes]
  q2 = q[NumModes:2*NumModes]
  dq11 = np.zeros((NumModes,NumModes))
  dq12 = np.zeros((NumModes,NumModes))
  dq21 = np.zeros((NumModes,NumModes))
  dq22 = np.zeros((NumModes,NumModes))
  for h in range(NumModes):
     for j in range(NumModes):

         for l in range(NumModes):
           dq11[j,h]=dq11[j,h]-(gamma1[j,h,l]+gamma1[j,l,h])*q1[l]
           dq12[j,h]=dq12[j,h]-(gamma2[j,h,l]*q2[l]+gamma2[j,l,h])*q2[l]
           dq21[j,h]=dq21[j,h]+gamma2[h,j,l]*q2[l]
           dq22[j,h]=dq22[j,h]+gamma2[l,j,h]*q1[l]

         dq11[j,h]=dq11[j,h]
         if j==h:
           dq12[j,h]=dq12[j,h]+Omega[j]
         else:
           dq12[j,h]=dq12[j,h]
         if j==h:
           dq21[j,h]=dq21[j,h]-Omega[j]
         else:
           dq21[j,h]=dq21[j,h]

         dq22[j,h]=dq22[j,h]


  return np.vstack((np.hstack((dq11,dq12)),np.hstack((dq21,dq22))))


def RK4(f,xn0,yn0,h,args1='None'):

    k1=f(xn0,yn0,args1)
    k2=f(xn0+0.5*h,yn0+0.5*k1*h,args1)
    k3=f(xn0+0.5*h,yn0+0.5*k2*h,args1)
    k4=f(xn0+h,yn0+k3*h,args1)

    yn1= yn0 + 1./6*(k1+2*k2+2*k3+k4)*h
    return(yn1)

def RK2(f,xn0,yn0,h,args1=()):

    k1=f(xn0,yn0,args1)
    k2=f(xn0+h/2,yn0+k1*h/2)
    yn1= yn0 + k2*h
    return(yn1)



'''


def Trapz():

def F():
 y-y0-h/2*(f(xn,yn,args1)+f(xn,yn,args1))

def JF():

h=(b-a)/N

while flag==0:

  ynew = yold - JF[xn0].dot(F(xn0))


'''


if (__name__ == '__main__'):

    print('Reading ODE')
