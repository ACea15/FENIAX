import numpy as np
import os
import pdb
import scipy
import scipy.linalg
import matplotlib.pyplot as plt



rhot = 0.#1.02 #kg/m3
# 1 ft = 0.3048 m ; 1 slug = 14.5939 Kg
rho = 0.0023760 #(rhot/14.5939)*0.3048**3
c=4.#ft
Amatrix = 'AERO/AICs00_4r3.npy'
Pvector = 'AERO/Poles00_4r3.npy'
AICs = np.load(Amatrix)
Poles =  np.load(Pvector)
# Amatrix = 'AERO/AICs4c_5r3.npy'
# Pvector = 'AERO/Poles4c_5r3.npy'
# AICs2 = np.load(Amatrix)
# Poles2 =  np.load(Pvector)
Ka = np.load('FEM/Kaa.npy')
Ma = np.load('FEM/Maa.npy')
Dreal,Vreal = scipy.linalg.eigh(Ka,Ma)
NumPoles = len(AICs)-3
NumModes = len(AICs[0])
Omega = np.sqrt(Dreal)[0:NumModes]
u_inf = 210.1

if NumPoles>0:
    Ap = np.hstack([AICs[i+3] for i in range(NumPoles)])

W = np.zeros((NumModes,NumModes))
for i in range(NumModes):
    W[i,i] = Omega[i]

def k(u):
    #u=557.
    return Omega*c/2/u


# if NumPoles>0:
#     def gamma(u):
#         gammax=np.zeros((NumPoles*NumModes,NumPoles*NumModes))
#         for i in range(NumPoles):
#             gammax[i*NumModes:(i+1)*NumModes,i*NumModes:(i+1)*NumModes] = k(u)[-1]/(i+1)*np.eye(NumModes)
#         return gammax

if NumPoles>0:
    def gamma(u):
        #u=557.
        gammax=np.zeros((NumPoles*NumModes,NumPoles*NumModes))
        for i in range(NumPoles):
            gammax[i*NumModes:(i+1)*NumModes,i*NumModes:(i+1)*NumModes] = Poles[i]*np.eye(NumModes)
        return gammax

def qinf(u):
    return 0.5*rho*u**2

def M1(uinf):
    #uinf=557.
    M = np.eye(NumModes)-qinf(uinf)*(c/(2*uinf))**2*AICs[2]
    Minv=np.linalg.inv(M)
    return Minv

def Ax(uinf):
    #uinf=557.
    A = np.zeros((3*NumModes+NumPoles*NumModes,3*NumModes+NumPoles*NumModes))

    A[0:NumModes,0:NumModes] = qinf(uinf)*c/(2*uinf)*M1(uinf).dot(AICs[1])
    A[0:NumModes,NumModes:2*NumModes] = M1(uinf).dot(W)
    A[0:NumModes,2*NumModes:3*NumModes] = qinf(uinf)*M1(uinf).dot(AICs[0])
    if NumPoles>0:
        A[0:NumModes,3*NumModes:3*NumModes+NumPoles*NumModes] = qinf(uinf)*M1(uinf).dot(Ap)

    A[NumModes:2*NumModes,0:NumModes] = -W
    A[2*NumModes:3*NumModes,0:NumModes] = np.eye(NumModes)
    if NumPoles>0:
        A[3*NumModes:3*NumModes+NumPoles*NumModes,0:NumModes] = np.vstack([np.eye(NumModes) for i in range(NumPoles)])
        A[3*NumModes:3*NumModes+NumPoles*NumModes,3*NumModes:3*NumModes+NumPoles*NumModes] = -2*uinf/c*gamma(uinf)

    return A


#At2=np.load('M600.npy')
At = Ax(u_inf)
D,V = scipy.linalg.eig(At)
print D#np.sqrt(D)

def plot_root(uf,NumModes):

    co = ['b','b', 'g','g', 'r', 'r','c','c', 'm','m', 'y','y', 'k','k']*NumModes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(uf)):

        Au = Ax(uf[i])
        D,V = scipy.linalg.eig(Au)
        dreal=np.real(D)[0:NumModes]
        dimag=np.imag(D)[0:NumModes]
        print dreal

        for j in range(NumModes):
            if i==0:
                figx = ax.plot(dreal[j],dimag[j],'X',color = co[j],label = 'Modes = %s' % j)
            else:
                figx = ax.plot(dreal[j],dimag[j],'X',color = co[j])

    plt.axvline(x=0.,linestyle='--',color='k')
    plt.legend()
    plt.show()


plot_root(range(50,220,50),10)
def qhh(q,u_inf):

    q1d = At.dot(q)[:NumModes]

    eta0 = 0.5*rho*(u_inf**2*AICs[0,:,:].dot(q[2*NumModes:3*NumModes]))
    print eta0
    eta1 = 0.5*rho*(u_inf*c/2*AICs[1,:,:].dot(q[:NumModes]))
    print eta1
    eta2 = 0.5*rho*((c/2)**2*AICs[2,:,:].dot(q1d))
    print eta2
    eta=  eta0+eta1+eta2
    # eta = 0.5*rho*(u_inf**2*AICs[0,:,:].dot(q[2*NumModes:3*NumModes]) +
    #                               u_inf*c/2*AICs[1,:,:].dot(q[:NumModes]) +
    #                               (c/2)**2*AICs[2,:,:].dot(q1d))
    for i in range(NumPoles):
        etai = 0.5*rho*u_inf**2*AICs[i+3,:,:].dot(q[(i+3)*NumModes:(i+4)*NumModes])
        print etai
        eta += etai
    return eta

def sys_sol(x0,ti,D,V):

    A0 = np.linalg.inv(V).dot(x0)
    u = np.ones((len(D),len(ti)))
    for i in range(len(D)):
        u[i,:] = D[i]*u[i,:]
    q = np.exp(u.dot(ti))
    return q

ti = np.linspace(0,50,10000)
x0 = np.zeros(len(D))
x0[1] = 0.2
x0[0] = 0.5
#D=D[0:2]
#V=V[:2,:2]
#x0=x0[0:2]
A0 = np.linalg.inv(V).dot(x0)
u = [[[] for j in range(len(ti))] for i in range(len(D))]#np.ones((len(D),len(ti)))
for i in range(len(D)):
    for j in range(len(ti)):
        u[i][j] = D[i]*ti[j]
u = np.asarray(u)
q = V.dot(np.diag(A0)).dot(np.exp(u))

#q = sys_sol(x0,ti,D,V)

#print qhh(1*np.ones(24),u_inf)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(q[0,:].real)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(q[0,:].real,q[1,:].real)
plt.show()



#plot_root([150,200,280,350,400,450,500],14)

#u_inf = 470
Phi0s = './NASTRAN/Phi1.npy'
Phi0 = np.load(Phi0s)
Mhh = Phi0.T.dot(Ma).dot(Phi0)
Khh = Phi0.T.dot(Ka).dot(Phi0)

Mh1 = np.linalg.inv(Mhh-0.5*rho*AICs[2]*(c/2.)**2)
AQ0 = Mh1.dot(qinf(u_inf)*AICs[0]-Khh)
AQ1 = 0.25*rho*c*u_inf*Mh1.dot(AICs[1])
AQp = qinf(u_inf)*Mh1.dot(Ap)
AQg = -2*u_inf/c*gamma(u_inf)



Aq0 = np.zeros((2*NumModes+NumPoles*NumModes,2*NumModes+NumPoles*NumModes))

Aq0[0:NumModes,NumModes:2*NumModes] = np.eye(NumModes)
Aq0[NumModes:2*NumModes,0:NumModes] = AQ0
Aq0[NumModes:2*NumModes,NumModes:2*NumModes] = AQ1
Aq0[NumModes:2*NumModes,2*NumModes:2*NumModes+NumPoles*NumModes] = AQp
Aq0[2*NumModes:2*NumModes+NumPoles*NumModes,NumModes:2*NumModes] = np.vstack([np.eye(NumModes) for i in range(NumPoles)])
Aq0[2*NumModes:2*NumModes+NumPoles*NumModes,2*NumModes:2*NumModes+NumPoles*NumModes] = AQg

D0,V0 = scipy.linalg.eig(Aq0)
print '---displacements---'
print D0


from Aerodynamics.rfa import RFA_freq


def k_method(kf):
    M1 = Mhh + 0.5*rho*(c/(2*kf))**2*RFA_freq(Poles,kf,AICs,'r')
    M2 = Khh
    D,V = scipy.linalg.eig(M1,M2)
    return D,V

kkm = 0.06991
Dkm,Vkm = k_method(kkm)
wkm=[]
gkm = []
vkm = []
print Dkm
for i in range(len(Dkm)):
    a=Dkm[i].real
    b=Dkm[i].imag
    #wkm.append(np.sqrt((a**2+b**2)/a))
    wkm.append(1./np.sqrt(a))
    gkm.append(b/a)
    vkm.append(wkm[i]*c/(2*kkm))


fnr = np.array([[-0.11220379, -1.06130038, -0.07435014],
       [ 0.17203856,  1.23838646,  0.02521833],
       [-0.05520362, -0.51516382, -0.0496446 ]])

fni = np.array([[-0.09804882, -0.17486293,  0.01429001],
       [ 0.03026051, -0.51629414, -0.07208212],
       [-0.04044014, -0.075806  , -0.13918649]])

u_inf = 542.
def pk_fun(kf):
    Qhhr = RFA_freq(Poles,kf,AICs,'r').real
    Qhhi = RFA_freq(Poles,kf,AICs,'r').imag
    Mhh1 = np.linalg.inv(Mhh)
    Akp = np.zeros((2*NumModes,2*NumModes))
    Akp[0:NumModes,NumModes:2*NumModes] = np.eye(NumModes)
    Akp[NumModes:2*NumModes,0:NumModes] = -Mhh1.dot(Khh -0.5*rho*u_inf**2*Qhhr)
    Akp[NumModes:2*NumModes,NumModes:2*NumModes] = Mhh1.dot(0.25*rho*c*u_inf*Qhhi/kf)

    return Akp


Omega = 18.74
#u_inf = 500
kf= Omega*c/(2*u_inf)
Akp = pk_fun(kf)
Dkp,Vkp = scipy.linalg.eig(Akp)
print '----PK Method----'
print Dkp

f2m=0.3048
mach = [0,0.3,0.5,0.7,0.8,0.9]
nastranpk = np.array([536.,532.,516.,479.,442.,383.])*f2m
intrinsic = np.array([544,541.3,533.5,509,482.5,429])*f2m
mach = [0,0.3,0.5,0.7]
nastranpk = np.array([536.,532.,516.,479.])*f2m
intrinsic = np.array([544,541.3,533.5,509])*f2m

fig = plt.figure(figsize=(8.,5.))
ax = fig.add_subplot(111)
ax.set_xlim(-0.1,0.8)
#ax.set_xlabel(-0.1,1)
ax.set_ylim(00,175)
ax.plot(mach,nastranpk,'xk',mfc='none',markersize=10,label='NASTRAN PK-results')
ax.plot(mach,intrinsic,'k--',linewidth=2,label='Intrinsic results')
ax.set_xlabel('Mach number')
ax.set_ylabel('Flutter speed (m/s)')
ax.legend(bbox_to_anchor=(1., 1.15), loc=1, borderaxespad=0.)
plt.grid(linestyle='dotted')
plt.show()


#################### Flutter ##############################
f2m=0.3048
f2m=1.
mach = [0,0.3,0.5,0.7,0.85]
nastranpk3d = np.array([209.5,208.7,205.2,196.2,179])*f2m
nastranpk3d = np.array([209.7,208.7,205.2,195.6,183])*f2m
nastranpk = np.array([208.1,207.4,203.6,194.4,181.9])*f2m
nastranpk = np.array([208.7,207.4,203.6,194.4,181.9])*f2m
intrinsic = np.array([209.3,208.1,204.7,195.5,181.2])*f2m
#mach = [0,0.3,0.5,0.7,0.85]
#nastranpk = np.array([536.,532.,516.,479.])*f2m
#intrinsic = np.array([544,541.3,533.5,509])*f2m

fig = plt.figure(figsize=(8.,5.))
ax = fig.add_subplot(111)
ax.set_xlim(-0.1,0.9)
#ax.set_xlabel(-0.1,1)
ax.set_ylim(160,225)
ax.plot(mach,nastranpk3d,'o',mfc='none',markersize=8,label='Full NASTRAN PK')
ax.plot(mach,nastranpk,'sk',mfc='none',markersize=8,label='ROM NASTRAN PK')
ax.plot(mach,intrinsic,'k.--',linewidth=2,label='NMROM')
ax.set_xlabel('Mach number')
ax.set_ylabel('Flutter speed (feet/s)')
ax.legend(bbox_to_anchor=(0.98, 0.95), loc=1, borderaxespad=0.)
plt.grid(linestyle='dotted')
save_flutter=1
save_name = 'flutter.pdf'
if save_flutter:
    plt.savefig(save_name,dpi=300, bbox_inches='tight')
    plt.close(fig)
plt.show()


#################### Frequencies ##########################


f2m=0.3048
f2m=1.
mach = [0,0.3,0.5,0.7,0.85]
nastranpk3d = np.array([9.19,8.76,8.06,7.11,6.32])*f2m
nastranpk = np.array([9.187,8.74,8.05,7.10,6.31])*f2m
intrinsic = np.array([9.2,8.75,8.07,7.1,6.3])*f2m
#mach = [0,0.3,0.5,0.7,0.85]
#nastranpk = np.array([536.,532.,516.,479.])*f2m
#intrinsic = np.array([544,541.3,533.5,509])*f2m

fig = plt.figure(figsize=(8.,5.))
ax = fig.add_subplot(111)
ax.set_xlim(-0.1,0.9)
#ax.set_xlabel(-0.1,1)
ax.set_ylim(5,10)
ax.plot(mach,nastranpk3d,'o',mfc='none',markersize=8,label='Full NASTRAN PK')
ax.plot(mach,nastranpk,'sk',mfc='none',markersize=8,label='ROM NASTRAN PK')
ax.plot(mach,intrinsic,'k.--',linewidth=2,label='NMROM')
ax.set_xlabel('Mach number')
ax.set_ylabel('Flutter frequency (rads/s)')
ax.legend(bbox_to_anchor=(0.98, 0.95), loc=1, borderaxespad=0.)
plt.grid(linestyle='dotted')
save_flutter=1
save_name = 'flutter_frequency.pdf'
if save_flutter:
    plt.savefig(save_name,dpi=300, bbox_inches='tight')
    plt.close(fig)
plt.show()

