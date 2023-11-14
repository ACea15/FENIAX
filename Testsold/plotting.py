import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import PostProcessing.plotting3 as plot
import Utils.op2reader as op2r
import pickle


scale=100./16
linewidth=2.5
markersize=6.6
color_internal=['k','steelblue']


def Rs(ra,ra0,tai,rn,tni,i,j,d,dis,xlim=None,ylim=None):

    fig = plt.figure()
    if dis:

        ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
        for ki in range(len(ra)):
            for di in d:
                figx = ax.plot(tai,ra[ki][i][j,:,di]-ra0[i][j,di],label = '%s BEAM = %s, POINT = %s, DIM = %s' % (ki,i,j,di))

        if rn is not []:
              k=BeamSeg[0][i]['NodeOrder'][j]
              for dn in d:
                  figx = ax.plot(tni,rn[:,k,dn]-rn[0,k,dn],'--')

    else:

        ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
        for ki in range(len(ra)):
            for di in d:
                figx = ax.plot(tai,ra[ki][i][j,:,di],label = '%s BEAM = %s, POINT = %s, DIM = %s' % (ki,i,j,di))

        if rn is not []:
              k=BeamSeg[0][i]['NodeOrder'][j]
              for dn in d:
                  figx = ax.plot(tni,rn[:,k,dn],'--')
    plt.xlim(xlim);plt.ylim(ylim)
    plt.legend()
    plt.show()

def Rs2(ra,ra0,tai,rn,tni,i,j,d,dis,xlim=None,ylim=None):

    fig = plt.figure()
    if dis:

        ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
        for ki in range(len(ra)):
            for di in d:
                figx = ax.plot(tai,ra[ki][i][j,:,di]-ra0[i][j,di],label = '%s BEAM = %s, POINT = %s, DIM = %s' % (ki,i,j,di))

        if rn is not []:
              k=BeamSeg[0][i]['NodeOrder'][j]
              for ki in range(len(rn)):
                  for dn in d:
                      figx = ax.plot(tni[ki],rn[ki][:,k,dn]-rn[ki][0,k,dn],'--')

    else:

        ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
        for ki in range(len(ra)):
            for di in d:
                figx = ax.plot(tai,ra[ki][i][j,:,di],label = '%s BEAM = %s, POINT = %s, DIM = %s' % (ki,i,j,di))

        if rn is not []:
              k=BeamSeg[0][i]['NodeOrder'][j]
              for ki in range(len(rn)):
                  for dn in d:
                      figx = ax.plot(tni[ki],rn[ki][:,k,dn],'--')
    plt.xlim(xlim);plt.ylim(ylim)
    plt.legend()
    plt.show()


def Rs3(ra,ra0,tai,rn,tni,i,j,d,dis,xlim=None,ylim=None,save=''):

    color = ['k','steelblue']
    colorn=['darkblue','grey']
    fig = plt.figure(figsize=(11,6))
    if dis:

        ax = fig.add_subplot(221)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
        ax.grid(linestyle='dotted')
        #ax.set_xlabel('t (sec.)',fontsize=12)
        ax.set_ylabel('$u_x$ (m)',fontsize=12)
        ax.set_xlim([0.,15.])
        for ki in range(len(ra)):
                di=0
                figx = ax.plot(tai,ra[ki][i][j,:,di]-ra0[i][j,di],c=color[ki])

        if rn is not []:
              k=BeamSeg[0][i]['NodeOrder'][j]
              for ki in range(len(rn)):
                  dn=0
                  figx = ax.plot(tni[ki],rn[ki][:,k,dn]-rn[ki][0,k,dn],'--',c=colorn[ki])

        ax = fig.add_subplot(223)
        ax.grid(linestyle='dotted')
        ax.set_xlabel('t (sec.)',fontsize=12)
        ax.set_ylabel('$u_y$ (m)',fontsize=12)
        ax.set_xlim([0.,15.])
        for ki in range(len(ra)):
                di=1
                figx = ax.plot(tai,ra[ki][i][j,:,di]-ra0[i][j,di],c=color[ki])

        if rn is not []:
              k=BeamSeg[0][i]['NodeOrder'][j]
              for ki in range(len(rn)):
                  dn=1
                  figx = ax.plot(tni[ki],rn[ki][:,k,dn]-rn[ki][0,k,dn],'--',c=colorn[ki])

        ax = fig.add_subplot(122)
        ax.grid(linestyle='dotted')
        ax.set_xlabel('t (sec.)',fontsize=12)
        ax.set_ylabel('$u_z$ (m)',fontsize=12)
        ax.set_xlim([0.,15.])
        label1= ['Kidder-reduced NMROM','Guyan-reduced NMROM']
        for ki in range(len(ra)):
                di=2
                figx = ax.plot(tai,ra[ki][i][j,:,di]-ra0[i][j,di],label = label1[ki],c=color[ki])
        labeln= ['NASTRAN NLin. Full FE','NASTRAN Lin. Full FE']
        if rn is not []:
              k=BeamSeg[0][i]['NodeOrder'][j]
              for ki in range(len(rn)):
                  dn=2
                  figx = ax.plot(tni[ki],rn[ki][:,k,dn]-rn[ki][0,k,dn],'--',label = labeln[ki],c=colorn[ki])

    plt.legend(bbox_to_anchor=(-0.3, 1.02, 1.3, 0.3), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.,fontsize=10)
    if save:
        fig.savefig(save, bbox_inches='tight')
        plt.close(fig)

    plt.show()



n1 = op2r.NastranReader('./nastran/wing_400b','./nastran/wing_400b')
n1.readModel()
tn,rn = n1.position()
un = n1.displacements()
n1l = op2r.NastranReader('./nastran/wing_109b','./nastran/wing_109b')
n1l.readModel()
tnl,rnl = n1l.position()
unl = n1l.displacements()
#tn = n1.op2.displacements[1].dts

p1g=plot.PlotX('./Results_modes/Geometry',None,'./F5e5g/Solv_53',tiX='./F5e5g/ti_53.npy')
p1g.readData()


p1bg=plot.PlotX('./Results_modes/Geometry',None,'./F5e5g/Solv_100',tiX='./F5e5g/ti_100.npy')
p1bg.readData()

p1i=plot.PlotX('./Results_modes/Geometry',None,'./F5e5i/Solv_53',tiX='./F5e5i/ti_53.npy')
p1i.readData()
p1bi=plot.PlotX('./Results_modes/Geometry',None,'./F5e5i/Solv_100',tiX='./F5e5i/ti_100.npy')
p1bi.readData()


with open ('./Results_modes/Geometry', 'rb') as fp:
    (BeamSeg) = pickle.load(fp)
Rs3([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1)
Rs([p1bi.ra,p1bg.ra],[p1bi.ra[0][:,0]],p1bg.tai,rn,tn,i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=[-0.8,0.2])
Rs([p1bi.ra,p1bg.ra],[p1bi.ra[0][:,0]],p1bg.tai,rn,tn,i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=[-2,0])
Rs([p1bi.ra,p1bg.ra],[p1bi.ra[0][:,0]],p1bg.tai,rn,tn,i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=[-8,8])

Rs2([p1bi.ra,p1bg.ra],[p1bi.ra[0][:,0]],p1bg.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=[-0.8,0.2])
Rs2([p1bi.ra,p1bg.ra],[p1bi.ra[0][:,0]],p1bg.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=[-2,0])
Rs2([p1bi.ra,p1bg.ra],[p1bi.ra[0][:,0]],p1bg.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=[-8,8])


n1 = op2r.NastranReader('./nastran2/wing400b','./nastran2/wing400b')
n1.readModel()
tn,rn = n1.position()
un = n1.displacements()
n1l = op2r.NastranReader('./nastran2/wing_109b','./nastran2/wing_109b')
n1l.readModel()
tnl,rnl = n1l.position()
unl = n1l.displacements()
#tn = n1.op2.displacements[1].dts


p1g=plot.PlotX('./Results_modes/Geometry',None,'./Fbg/Solv_53',tiX='./Fbg/ti_53.npy')
p1g.readData()
# p1bg=plot.PlotX('./Results_modes/Geometry',None,'./Fbg/Solv_100',tiX='./Fbg/ti_100.npy')
# p1bg.readData()

p1i=plot.PlotX('./Results_modes/Geometry',None,'./Fbi/Solv_53',tiX='./Fbi/ti_53.npy')
p1i.readData()
# p1bi=plot.PlotX('./Results_modes/Geometry',None,'./F5e5i/Solv_100',tiX='./F5e5i/ti_100.npy')
# p1bi.readData()


with open ('./Results_modes/Geometry', 'rb') as fp:
    (BeamSeg) = pickle.load(fp)

# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rn,tn,i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=None)
# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rn,tn,i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=None)
# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rn,tn,i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=None)
Rs3([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=None)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=None)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=None)




n1 = op2r.NastranReader('/media/pcloud/Computations/FEM4INAS/Models/wingSP/NASTRAN/calcnew/angle/old/wing400d','/media/pcloud/Computations/FEM4INAS/Models/wingSP/NASTRAN/calcnew/angle/old/wing400d')
n1.readModel()
tn,rn = n1.position()

un = n1.displacements()
n1l = op2r.NastranReader('./nastran2/wing_109c','./nastran2/wing_109b')
n1l.readModel()
tnl,rnl = n1l.position()
unl = n1l.displacements()
#tn = n1.op2.displacements[1].dts

p1g=plot.PlotX('./Results_modes/Geometry',None,'./Fcg/Solv_53',tiX='./Fcg/ti_53.npy')
p1g.readData()
# p1bg=plot.PlotX('./Results_modes/Geometry',None,'./Fbg/Solv_100',tiX='./Fbg/ti_100.npy')
# p1bg.readData()

p1i=plot.PlotX('./Results_modes/Geometry',None,'./Fci/Solv_53',tiX='./Fci/ti_53.npy')
p1i.readData()
# p1bi=plot.PlotX('./Results_modes/Geometry',None,'./F5e5i/Solv_100',tiX='./F5e5i/ti_100.npy')
# p1bi.readData()


with open ('./Results_modes/Geometry', 'rb') as fp:
    (BeamSeg) = pickle.load(fp)

# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rn,tn,i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=None)
# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rn,tn,i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=None)
# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rnl,tnl,i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=None)
Rs3([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=None)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=None)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=None)

#running d #############################################################################


with open ('/media/pcloud/Computations/FEM4INAS/Models/wingSP/Results_modes/Geometry', 'rb') as fp:
    (BeamSeg) = pickle.load(fp)


rn = np.load('./DynamicData/wingSP_re.npy')
un = np.load('./DynamicData/wingSP_ue.npy')
tn = np.load('./DynamicData/wingSP_te.npy')

p1g2=plot.PlotX('/media/pcloud/Computations/FEM4INAS/Models/wingSP/Results_modes/Geometry',None,'/media/pcloud/Computations/FEM4INAS/Models/wingSP/Solv_53g',tiX='./DynamicData/ti_53.npy')
p1g2.readData()
p1i2=plot.PlotX('/media/pcloud/Computations/FEM4INAS/Models/wingSP/Results_modes/Geometry',None,'./DynamicData/Solv_53i',tiX='./DynamicData/ti_53.npy')
p1i2.readData()

p1g=plot.PlotX('/media/pcloud/Computations/FEM4INAS/Models/wingSP/Results_modes/Geometry',None,'/media/pcloud/Computations/FEM4INAS/Models/wingSP/Feg/Solv_53',tiX='./DynamicData/ti_53.npy')
p1g.readData()
p1i=plot.PlotX('/media/pcloud/Computations/FEM4INAS/Models/wingSP/Results_modes/Geometry',None,'/media/pcloud/Computations/FEM4INAS/Models/wingSP/Feg/Solv_53',tiX='./DynamicData/ti_53.npy')
p1i.readData()


def error(ra,rn):
    tna = len(ra[0][0])
    tnn = len(rn)
    sum=0.
    for i in range((tnn-1)/2):
        sum+=np.linalg(rn[2*i][-4]-ra[0][-1,3*i])/np.linalg(rn[2*i][-4])
    sum=sum/(tnn-1)/2
    return sum

        
with open ('./Results_modes/Geometry', 'rb') as fp:
    (BeamSeg) = pickle.load(fp)

# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rnl,tnl,i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=None)
# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rnl,tnl,i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=None)
# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rnl,tnl,i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=None)
Rs3([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1)


Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=None)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=None)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=None)



def point_outside(ra,R,arm,point1):

    n = len(ra[0][-1])
    r = np.zeros((n,3))
    u = np.zeros((n,3))
    for i in range(n):
        u[i]=ra[0][-1][i]+R[0][-1][i].dot(arm)-point1
        r[i]=ra[0][-1][i]+R[0][-1][i].dot(arm)
        #rin[i]=ra[0][-1][i]

    return r,u


def angle(x1,x2):

    angle_n = np.zeros(len(x1))
    for i in range(len(x1)):
        angle_n[i] = np.arcsin((x2[i]-x1[i])[2]/np.linalg.norm(x2[i]-x1[i]))
    return angle_n

nas1 = op2r.NastranReader('./nastran/angle/wing400e','./nastran2/wing400b')
nas1.readModel()
tnas,rnas = nas1.position()
unas = nas1.displacements()
nas1l = op2r.NastranReader('./nastran/angle/wing_109e','./nastran2/wing_109b')
nas1l.readModel()
tnasl,rnasl = nas1l.position()
unasl = nas1l.displacements()
#tn = n1.op2.displacements[1].dts




angle_nastran = angle(rnas[:,0],rnas[:,-4])
angle_nastranl = angle(rnasl[:,0],rnasl[:,-4])
point1=np.array([19.232,28.8,-0.37977])
arm = point1-p1i.ra[0][-1][00]
arm=p1i.Rab[0][-1][000].T.dot(arm)
r_outside,u_outside =  point_outside(p1i.ra,p1i.Rab,arm,point1)
angle_r = angle(p1i.ra[0][-1],r_outside)


fig = plt.figure()
ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
figx = ax.plot(p1g.tai,angle_r,label = 'NMROM')
figx = ax.plot(tnas,-angle_nastran,'--',label='Nastran Nonlinear')
figx = ax.plot(tnas,-angle_nastranl,'r',label = 'Nastran Linear')
plt.xlim([0,15]);plt.ylim([-0.2,0.25])
plt.legend()
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
#figx = ax.plot(p1g.tai[0::3],360./2/np.pi*(angle_r[0::3]+angle_nastran[0::2]),label = 'NMROM')
#figx = ax.plot(tnas,360./2/np.pi*-angle_nastran,'--',label='Nastran Nonlinear')
figx = ax.plot(tnas,360./2/np.pi*(angle_nastranl-angle_nastran),'r',label = 'Nastran Linear')
#plt.xlim([0,15]);plt.ylim([-0.2,0.25])
plt.legend()
plt.show()


#360./2/np.pi*














n1 = op2r.NastranReader('./nastran2/wing400e','./nastran2/wing400b')
n1.readModel()
tn,rn = n1.position()
un = n1.displacements()
n1l = op2r.NastranReader('./nastran2/wing_109e','./nastran2/wing_109b')
n1l.readModel()
tnl,rnl = n1l.position()
unl = n1l.displacements()
#tn = n1.op2.displacements[1].dts

p1g=plot.PlotX('./Results_modes/Geometry',None,'./Feg/Solv_53',tiX='./Feg/ti_53.npy')
p1g.readData()
# p1bg=plot.PlotX('./Results_modes/Geometry',None,'./Fbg/Solv_100',tiX='./Fbg/ti_100.npy')
# p1bg.readData()

p1i=plot.PlotX('./Results_modes/Geometry',None,'./Fei/Solv_53',tiX='./Fei/ti_53.npy')
p1i.readData()
# p1bi=plot.PlotX('./Results_modes/Geometry',None,'./F5e5i/Solv_100',tiX='./F5e5i/ti_100.npy')
# p1bi.readData()


with open ('./Results_modes/Geometry', 'rb') as fp:
    (BeamSeg) = pickle.load(fp)

# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rn,tn,i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=None)
# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rn,tn,i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=None)
# Rs([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,rn,tn,i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=None)

Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[0],dis=1,xlim=[0,10],ylim=None)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[1],dis=1,xlim=[0,10],ylim=None)
Rs2([p1i.ra,p1g.ra],[p1i.ra[0][:,0]],p1g.tai,[rn,rnl],[tn,tnl],i=0,j=-1,d=[2],dis=1,xlim=[0,10],ylim=None)



Ka = np.abs(np.load('./FEM/K.npy'))
#fig, ax = plt.subplots()
#plt.matshow(Ka, cmap=plt.cm.Blues)
plt.matshow(Ka, vmax=np.max(Ka)/1000)
plt.show()

#p2.Animation2D([1,2],interval=1,Axes=[[0,16],[-0.5,8.5]],save=1,save_name='Hale_4Deg.mp4')
#p1.Animation2D([1,2],interval=1,Axes=[[0,16],[-0.5,4]],save=1,save_name='Hale_2Deg.mp4')
