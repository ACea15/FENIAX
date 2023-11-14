import numpy as np

def err_gamma1(X1,gamma1,gamma1lT):

    err1=[]

    for i in range(len(X1)):
        g1=gamma1[X1[i][0],X1[i][1],X1[i][2]]
        g1t=gamma1lT[i]
        err1.append(abs(abs(g1)-abs(g1t)))

    err1r=[]

    for i in range(len(X1)):
        g1r=gamma1[X1[i][0],X1[i][1],X1[i][2]]
        g1tr=gamma1lT[i]
        err1r.append(abs(abs(g1r)-abs(g1tr))/abs(g1tr))

    return max(err1),max(err1r)

def err_gamma2(X2,gamma2,gamma2lT):

    err2=[]
    for i in range(len(X2)):
        g2=gamma2[X2[i][0],X2[i][1],X2[i][2]]
        g2t=gamma2lT[i]
        err2.append(abs(abs(g2)-abs(g2t)))

    err2r=[]
    for i in range(len(X2)):
        g2r=gamma2[X2[i][0],X2[i][1],X2[i][2]]
        g2tr=gamma2lT[i]
        err2r.append(abs(abs(g2r)-abs(g2tr))/abs(g2tr))


    return max(err2),max(err2r)


def reading_gamma(variable):
    a1=[]
    a2=[]
    with open(variable) as f:
        for line in f:
            data = line.split()
            a1.append([int(float(data[0])-1),int(float(data[1])-1),int(float(data[2])-1)])
            a2.append(float(data[3]))

    return a1,a2


def err_static_tip(ra,raC):
      err=[]
      for j in range(len(ra[0][-1])):
          # print ra[i][-1][j]
          # print raC[i][-1][j]
          # print '###'
          err.append(np.linalg.norm(ra[-1][-1,j] - raC[-1][-1,j])/np.linalg.norm(raC[-1][-1,j]))
      return err


def err_static_tip_simo45(ra,raC):
      err=[]
      for j in range(len(ra[0][-1])):
          #print ra[-1][-1][j]
          #print raC[-1][-1][j]
          #print '###'
          #print np.shape(ra)
          #print np.shape(raC)
          err.append(np.linalg.norm(ra[-1][-1,j] - raC[-1][-1,j*3])/np.linalg.norm(raC[-1][-1,j*3]))
      return err


def err_norm(ra,raC,beams):
    sum = 0.
    for i in range(len(beams)):
        for f in range(len(ra[i][0])):
            sum += np.linalg.norm(ra[i][:,f] - raC[i][:,f])/np.sqrt(len(ra[i]))
        sum /= len(ra[i][0])
    sum /= len(beams)
    return sum

def err_rafabeam(feminas,nastran,points):

    ln = len(points)
    tn= len(nastran)
    Sx=np.zeros(ln);Sy=np.zeros(ln);Sz=np.zeros(ln)
    for i in range(ln):
        vec1 = feminas[0][points[i]]
        vec2 = nastran[:,points[i]]
        sx=0.;sy=0.;sz=0.
        for ti in range(tn):
            if abs(vec2[ti,0])>1e-3:
                sx += abs(vec1[ti*2,0]-vec2[ti,0])/vec2[ti,0]
            if abs(vec2[ti,1])>1e-3:
                sy += abs(vec1[ti*2,1]-vec2[ti,1])/vec2[ti,1]
            if abs(vec2[ti,2])>1e-3:
                sz += abs(vec1[ti*2,2]-vec2[ti,2])/vec2[ti,2]
        Sx[i] = sx/tn ; Sy[i] = sy/tn ; Sz[i] = sz/tn

    return [Sx,Sy,Sz]



def Hesse25_cgx(t):

  k0=2./2.5
  k1=-2./2.5
  k2 = 0.

  s0 = 0.
  v00=0.

  v10=v00+k0*(2.5)**2/2
  v20=v10 + 2*(5.-2.5)+k1*(5.-2.5)**2/2

  s10 = s0+v00*2.5+k0*2.5**3/6
  s20 = s10+v10*(5.-2.5)+(5.-2.5)**2+k1*(5-2.5)**3/6


  if t<2.5:
    x1 = s0+v00*t+k0*t**3/6
    return x1

  elif t<5.:

    x2 = s10+v10*(t-2.5)+(t-2.5)**2+k1*(t-2.5)**3/6
    return x2

  else:
    x3 = s20+v20*(t-5.)+k2*(t-5.)**3/6
    return x3

def Hesse25_cgx2d(t):

  a0 = 8./10.  
  s0 = 0.
  v00=0.
  v10=v00+a0*(2.5)
  s10 = s0+v00*2.5+a0*2.5**2/2

  if t<2.5:
    x1 = s0+v00*t+a0*t**2/2
    return x1

  else:

    x2 = s10+v10*(t-2.5)
    return x2

def Hesse25_cg2d(t):
    return np.array([Hesse25_cgx2d(t)+3,-4.,0.])

def Hesse25_cgerr2d(cga,ti,tl=None):

    if tl is not None:
        tlim=tl
    else:
        tlim = ti[-1]
    ni=0
    err=0.
    for i in range(len(ti)):
        if ti[i]<=tlim:
            err+=(np.linalg.norm(cga[i]-Hesse25_cg2d(ti[i])))**2
            ni+=1

    Err=np.sqrt(err/ni)/np.linalg.norm(Hesse25_cg2d(0))
    return Err

def Hesse25_cg(t):
    return np.array([Hesse25_cgx(t)+3,-4.,0.])

def Hesse25_cgerr3d(cga,ti,tl=None):

    if tl is not None:
        tlim=tl
    else:
        tlim = ti[-1]
    ni=0
    err=0.
    for i in range(len(ti)):
        if ti[i]<=tlim:
            err+=(np.linalg.norm(cga[i]-Hesse25_cg(ti[i])))**2
            ni+=1

    Err=np.sqrt(err/ni)/np.linalg.norm(Hesse25_cg(0))
    return Err


def err_SailPlane_static(ra,rn,BeamSeg):
    err=[]
    err2=[]
    for fi in range(len(rn)):
        erri=0.
        erri2=0.
        count=0
        for i in range(len(ra)):
            erri += np.linalg.norm(ra[i][1:,fi]-rn[fi][BeamSeg[i]['NodeOrder'][1:]])/np.linalg.norm(ra[i][1:,fi])
            erri2 += np.linalg.norm(ra[i][1:,fi]-rn[fi][BeamSeg[i]['NodeOrder'][1:]])
            count+=1
        err.append(erri/count)
        err2.append(erri2/count)
    return err,err2

def err_DoublePendulum(rth,ra):

    sum = 0.
    for i in range(len(rth)):
            sum += np.linalg.norm((rth[i][1:]-ra[1][0][:,i]))/np.linalg.norm((ra[1][0][:,i]))
    sum /= len(rth)
    return sum

def err_DoublePendulumFixed(rth,ra):
    sum = 0.
    for i in range(len(rth)):
            sum += np.linalg.norm((rth[i][1:3]-ra[1][0][:,i]))/np.linalg.norm((ra[1][0][:,i]))
    sum /= len(rth)
    return sum


def err_SailPlaneWing(ra,rn):
    tna = len(ra[0][0])
    tnn = len(rn)
    sum=0.
    for i in range((tnn-1)/2):
        sum+=np.linalg.norm(rn[2*i][-1]-ra[0][-1,3*i])/np.linalg.norm(rn[2*i][-1])
    sum=sum/(tnn-1)/2
    return sum

