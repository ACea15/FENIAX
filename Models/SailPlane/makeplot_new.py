import numpy as np
import pdb
import sys
import os
import pickle, gzip
import importlib
import  PostProcessing.plotting3 as pp
#reload(IntrinsicSolver.Tools.plotting)
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import intrinsic.geometryrb
from Utils.common import class2dic
V = importlib.import_module("Runs"+'.'+'SailPlane.V')
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn = intrinsic.geometryrb.geometry_def(V.Grid,
                                                  V.NumBeams,V.BeamConn,V.start_reading,
                                                  V.beam_start,V.nodeorder_start,V.node_start,                                                  V.Clamped,V.ClampX,V.BeamsClamped,V.MBbeams)

# with open('./Results_modes'+'/Geometry', 'wb') as fp:
#    pickle.dump(([class2dic(BeamSeg[i]) for i in range(len(BeamSeg))],NumNode, NumNodes, DupNodes, inverseconn),fp)


##########################
# Static Plot

sp2=pp.PlotX(solX='./Test/Results_FRES/Sol_50',geometryX='./Results_modes/Geometry',nastranX='./SP400/rt.npy')
sp2.readData()

sp1=pp.PlotX(solX='./results_static/Sols_50',geometryX='./Results_modes/Geometry',nastranX='./SP400/rt.npy')
sp1.readData()
sp1.Static3DDisp([0,1,2,3,4,5],ClampX=BeamSeg[0].NodeX[0],BeamsClamped=[0,1],nastran=range(6))


def err_SailPlane_static(ra,rn,BeamSeg):
    err=[]
    err2=[]
    for fi in range(len(rn)):
        erri=0.
        erri2=0.
        count=0
        for i in range(len(ra)):
            erri += np.linalg.norm(sp1.ra[i][1:,fi]-sp1.rn[fi][BeamSeg[i].NodeOrder[1:]])/np.linalg.norm(sp1.ra[i][1:,fi])
            erri2 += np.linalg.norm(sp1.ra[i][1:,fi]-sp1.rn[fi][BeamSeg[i].NodeOrder[1:]])
            count+=1
        err.append(erri/count)
        err2.append(erri2/count)
    return err,err2

sp_static=sail_plane('Test/Results_FRES','Test/Results_modes',results_nastran='SP400',nastrandata={'r':'/rt.npy','u':'/ut.npy'})
sp_static.V.NumModes = 50
sp_static.readData(local=0)

global legend1,save_static,static_save,dpi_static
save_static =0
dpi_static = 250
static_save = 'Figures/static.pdf'
legend1=[0.3,2.15]
sp_static.Static3DDisp([0,1,2,3,4,5],nastran=[0,1,2,3,4,5])



##########################
# Modes Plot
sp_static.mode_disp2([1,2,4],0,angular=0,Axes=[None,None,None])

global view,save_modes,modes_save
view=[[18,-178],[18,-178],[19,-165],[17,-160]]
view=[[18,-178],[18,-178],[62,-178],[62,-178]]
view=[[8,-171],[68,-179],[6,-172],[62,-178]]
save_modes=0
modes_save = 'Figures/modes135_1.pdf'
sp_static.mode_plot([6,8,9],2,angular=0,Axes=[[[7,45],[-30,30],[-12,12]],[[7,45],[-30,30],[-4,12]],[[0,45],[-30,30],[-2,12]],[[7,45],[-30,30],[-2,12]]])

sp_static.mode_plot([1,2,3],2,angular=0,Axes=[[[7,45],[-30,30],[-12,12]],[[7,45],[-30,30],[-4,12]],[[0,45],[-30,30],[-2,12]],[[7,45],[-30,30],[-2,12]]])

sp_static.mode_plot([1,2,4],1,angular=0,Axes=[[[16,40],[-30,30],[-12,16]],[[0,40],[-30,30],[-4,16]],[[0,45],[-30,30],[-12,16]],[[7,45],[-30,30],[-12,16]]])


###############################
# Dynamic Plot

sp_dyn=sail_plane('Test/Results_F0_60-40','Test/Results_modes',results_nastran='NastranResults/SP-60-40',nastrandata={'r':'/r.npy','u':'/u.npy'})
sp_dyn.V.NumModes = 90
sp_dyn.readData()

sp_dyn.Rt(i=6,j=-1,d=1,dis=1,nastran=1,external=0)

sp_dyn.Animation3D(1,[[0,45],[-35,35],[-20,20]])

# Dynamic linear

sp_dynl=sail_plane('Test/Results_F0_3-2','Test/Results_modes',results_nastran='NastranResults/SP-3-2',nastrandata={'r':'/r.npy','u':'/u.npy'})
sp_dynl.V.NumModes = 90
sp_dynl.readData()

sp_dynl.Rt(i=6,j=-1,d=2,dis=1,nastran=1,external=0)
###########################################################
global save_q,q_save
save_q=1
q_save='Figures/sp_q2.pdf'
qplot(sp_dyn,range(10))

###########################################################
# Energy computations
save_energy = 0
energy_save = 'Figures/sp_energy.pdf'
def e0(NumModes,q1,q2):
    tn=len(q1)
    E0=np.zeros(tn)
    E1=np.zeros(tn)
    E2=np.zeros(tn)
    for i in range(tn):
        for m in range(NumModes):
            E0[i]=E0[i]+0.5*(q1[i,m]**2+q2[i,m]**2)
            E1[i]=E1[i]+0.5*(q1[i,m]**2)
            E2[i]=E2[i]+0.5*(q2[i,m]**2)
    return E0,E1,E2

E0,E1,E2 = e0(90,sp_dyn.q[:,:90],sp_dyn.q[:,90:])
E0l,E1l,E2l = e0(90,sp_dynl.q[:,:90],sp_dynl.q[:,90:])


fig = plt.figure(figsize=(6.5,3.25))
ax = fig.add_subplot(211)
figx = ax.plot(sp_dyn.tai,E0/E0[0],'k--',linewidth=1.5,label='Total energy, $E_0$')
figx = ax.plot(sp_dyn.tai,E1/E0[0],c='grey',linewidth=1,label='Kinematic energy, $E_k$')
figx = ax.plot(sp_dyn.tai,E2/E0[0],c='steelblue',linewidth=1,label='Potential energy, $E_p$')
#plt.show()
ax.set_xlim([-0.01,5.])
ax.set_ylim([0.,1.05])
ax.set_ylabel('$E_i/E_0(0)$')
#ax.grid(linestyle='dotted')
#fig = plt.figure()
ax2 = fig.add_subplot(212)
figx = ax2.plot(sp_dyn.tai,E0l/E0l[0],'k--')
figx = ax2.plot(sp_dyn.tai,E1l/E0l[0],c='grey',linewidth=1)
figx = ax2.plot(sp_dyn.tai,E2l/E0l[0],c='steelblue',linewidth=1)
ax2.set_xlim([-0.01,5.])
ax2.set_ylim([0,1.05])
ax2.set_xlabel('Time (secs.)')
ax2.set_ylabel('$E_i/E_0(0)$ (Linear)')
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
fig.tight_layout()

if save_energy:
    plt.savefig(energy_save,dpi=250)
else:
    plt.show()






























# sp_static=IntrinsicSolver.Tools.plotting.PlotX('SailPlaneCnew','Test/Results_FRES','Test/Results_modes',results_nastran='SP400',nastrandata={'r':'/ut.npy'})
# sp_static.V.NumModes = 50
# sp_static.readData()
# sp_static.Static3DDisp([0,1,2,3,4,5],nastran=[0,1,2,3,4])

sp_dyn=IntrinsicSolver.Tools.plotting.PlotX('SailPlaneCnew','Test/Results_F0_60-40','Test/Results_modes',results_nastran='NastranResults/SP-60-40',nastrandata={'r':'/r.npy','u':'/u.npy'})
sp_dyn.V.NumModes = 90
sp_dyn.readData()

sp_dyn.Rt(6,-1,2,0,1,0)

sp_dynl=IntrinsicSolver.Tools.plotting.PlotX('SailPlaneCnew','Test/Results_F0_3-2','Test/Results_modes',results_nastran='NastranResults/SP-3-2',nastrandata={'r':'/r.npy','u':'/u.npy'})
sp_dynl.V.NumModes = 90
sp_dynl.readData()

sp_dynl.Rt(6,-1,0,0,1,0)


sp_staticl=sail_plane('Test/Results_FRES','Test/Results_modes',results_nastran='SP400',nastrandata={'r':'/rl.npy','u':'/ul.npy'})
sp_staticl.V.NumModes = 50
sp_staticl.readData(local=0)
sp_staticl.Static3DDisp([0,1,2,3,4,5],nastran=[0,1,2,3,4,5])

def arclength(ra,ra0):

    na = np.shape(ra)[0]
    tn = np.shape(ra)[1]
    st = np.zeros(tn)
    st0=0.
    for i in range(na-1):
        st0 += np.linalg.norm(ra0[i+1] - ra0[i])

    for ti in range(tn):
        for i in range(na-1):
            st[ti] += np.linalg.norm(ra[i+1][ti] - ra[i][ti])

    return st0,st/st0


st0l,stl = arclength(sp_dynl.ra[6])
st0,st = arclength(sp_dyn.ra[6])



def arclengthn(sp,b):

    na = len(sp.BeamSeg[b].NodeOrder)-1
    k = sp.BeamSeg[b].NodeOrder[1:]
    tn = np.shape(sp.rn)[0]
    st = np.zeros(tn)
    st0=0.
    for i in range(na-1):
        st0 += np.linalg.norm(sp.rn[0][k[i+1]][0:3] - sp.rn[0][k[i]][0:3])

    for ti in range(tn):
        for i in range(na-1):
            st[ti] += np.linalg.norm(sp.rn[ti][k[i+1]][0:3] - sp.rn[ti][k[i]][0:3])

    return st0,st/st0



st0l2,stl2 = arclengthn(sp_dynl,6)
st02,st2 = arclengthn(sp_dyn,6)

sts0,sts = arclength(sp_static.ra[6],sp_static.ra0[6])
sts02,sts2 = arclengthn(sp_static,6)


sts03,sts3 = arclength(sp_static.ra[6][1:],sp_static.ra0[6][1:])
sts0l,stsl = arclengthn(sp_staticl,6)

axil_static=(sts-1)*100
axil_staticl=(sts0l*stsl/sts03-1)*100

F = [200,250,300,400,480,530]
Axs = np.hstack([axil_staticl])
Axsl = np.hstack([axil_static])

Rst=[[] for i in range(6)]
Rstl=[[] for i in range(6)]
Rstn=[[] for i in range(6)]
Xst=[[] for i in range(6)]
Xstl=[[] for i in range(6)]
Xstn=[[] for i in range(6)]
Yst=[[] for i in range(6)]
Ystl=[[] for i in range(6)]
Ystn=[[] for i in range(6)]

for i in range(6):
    Rst[i]=np.vstack([sp_static.ra[2][:,i],sp_static.ra[3][1:,i],sp_static.ra[4][1:,i],sp_static.ra[5][1:,i],sp_static.ra[6][1:,i]])
    Rstl[i]=np.vstack([sp_staticl.rn[i,sp_staticl.BeamSeg[2].NodeOrder[1:]],
                   sp_staticl.rn[i,sp_staticl.BeamSeg[3].NodeOrder[1:]],
                   sp_staticl.rn[i,sp_staticl.BeamSeg[4].NodeOrder[1:]],
                   sp_staticl.rn[i,sp_staticl.BeamSeg[5].NodeOrder[1:]],
                    sp_staticl.rn[i,sp_staticl.BeamSeg[6].NodeOrder[1:]]])
    Rstn[i]=np.vstack([sp_static.rn[i,sp_static.BeamSeg[2].NodeOrder[1:]],
                   sp_static.rn[i,sp_static.BeamSeg[3].NodeOrder[1:]],
                   sp_static.rn[i,sp_static.BeamSeg[4].NodeOrder[1:]],
                   sp_static.rn[i,sp_static.BeamSeg[5].NodeOrder[1:]],
                    sp_static.rn[i,sp_static.BeamSeg[6].NodeOrder[1:]]])

    Xst[i]=np.sqrt(Rst[i][:,0]**2+Rst[i][:,1]**2)
    Xstl[i]=np.sqrt(Rstl[i][:,0]**2+Rstl[i][:,1]**2)
    Xstn[i]=np.sqrt(Rstn[i][:,0]**2+Rstn[i][:,1]**2)
    Yst[i]=Rst[i][:,2]
    Ystl[i]=Rstl[i][:,2]
    Ystn[i]=Rstn[i][:,2]


axial_save = 'Figures/sp_axial.pdf'
save_axial=0
fig = plt.figure(figsize=(5,3.5))
ax = fig.add_subplot(111)
figx = ax.plot(F,Axs,'k-o',label='NASTRAN linear Sol.')
figx = ax.plot(F,Axsl,color='steelblue',marker='o',label='NMROM')
ax.set_xticks(F)
#ax.set_yticks(range(8))
#ax.set_xlim([-0.01,5.])
#ax.set_ylim([0.,1.05])
ax.set_ylabel('$\Delta$ Axial length (%)',fontsize=12)
ax.set_xlabel('F (KN)',fontsize=12)
ax.tick_params(labelsize=10)
ax.grid(linestyle='dotted')
ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.,fontsize=10)
fig.tight_layout()
if save_axial:
    plt.savefig(axial_save,dpi=300)
else:
    plt.show()

axial_save2 = 'Figures/sp_axial2.pdf'
save_axial2=0
fig = plt.figure(figsize=(6,3.5))
ax2 = fig.add_subplot(111)
for i in range(6):

    if i==0:
        figx = ax2.plot(Xst[i],Yst[i],color='steelblue',linewidth=2.2,label='NMROM')
        figx = ax2.plot(Xstl[i],Ystl[i],'k',label='NASTRAN Lin. Sol.')
        figx = ax2.plot(Xstn[i],Ystn[i],'k--',label='NASTRAN NL Sol.')
    figx = ax2.plot(Xst[i],Yst[i],color='steelblue',linewidth=1.9)
    figx = ax2.plot(Xstl[i],Ystl[i],'k')
    figx = ax2.plot(Xstn[i],Ystn[i],'k--')
#ax2.set_xlim([-0.01,5.])
#ax2.set_ylim([0,1.05])
ax2.set_xlabel('$\sqrt{X_a^2 + Y_a^2}$',fontsize=13)
ax2.set_ylabel('$Z_a$',fontsize=13)
ax2.tick_params(labelsize=10)
ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.,fontsize=10)
ax2.grid(linestyle='dotted')
fig.tight_layout()
if save_axial2:
    plt.savefig(axial_save2,dpi=300)
else:
    plt.show()



def fc(x):
    return int(float(tn-1)/tf*x)

class sail_plane(IntrinsicSolver.Tools.plotting.PlotX):

    def __init__(self,results,results_modes,results_nastran,nastrandata,labelx=None):
         IntrinsicSolver.Tools.plotting.PlotX.__init__(self,'SailPlaneCnew',results,results_modes,results_nastran,nastrandata)
         self.labelx=labelx


    def Static3DDisp(self,rt,nastran=[],axi=None):

      fig = plt.figure(figsize=(7.5,6.5))
      #fig.subplots_adjust(left=-3)
      ax = fig.add_subplot(211, projection='3d')
      ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
      ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
      ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

      for ti in rt:

        for i in range(self.V.NumBeams):
              if ti==rt[0]:
                  x = self.BeamSeg[i].NodeX[:,0]
                  y = self.BeamSeg[i].NodeX[:,1]
                  z = self.BeamSeg[i].NodeX[:,2]
                  if i==0:
                      ax.plot(x, y, z,c='k', marker='o',markersize=3,label='Initial Configuration')
                  else:
                      ax.plot(x, y, z, c='k', marker='o',markersize=3)
              #rerr = self.ra0[i]-self.BeamSeg[i].NodeX
              rx= self.ra[i][:,ti,0]#-rerr[:,0]
              ry= self.ra[i][:,ti,1]#-rerr[:,1]
              rz= self.ra[i][:,ti,2]#-rerr[:,2]
              #ax.scatter(x, y, z, c='r', marker='o')
              if i==0 and ti==rt[0]:
                  ax.plot(rx,ry,rz,c='steelblue',linewidth=1.8,label='Intrinsic Theory')
              else:
                  ax.plot(rx,ry,rz,c='steelblue',linewidth=1.8)

        # if self.results_nastran:
        #     for ni in nastran:
        #             figx = ax.plot(self.rn[ni,:,0],self.rn[ni,:,1],self.rn[ni,:,2],'--')
      for ni in nastran:
            for i in range(self.V.NumBeams):
                if i in self.V.BeamsClamped:
                    #pdb.set_trace()
                    sx = np.hstack((self.V.ClampX[0],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],0]))
                    sy = np.hstack((self.V.ClampX[1],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],1]))
                    sz = np.hstack((self.V.ClampX[2],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],2]))

                else:
                    sx = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]].NodeOrder[-1],0],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],0]))
                    sy = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]].NodeOrder[-1],1],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],1]))
                    sz = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]].NodeOrder[-1],2],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],2]))
                if i==0 and ni==nastran[0]:

                    figx = ax.plot(sx,sy,sz,'k--',linewidth=1.8,label='Nastran Nonlinear')
                else:
                    figx = ax.plot(sx,sy,sz,'k--',linewidth=1.8)

      #ax.grid(False)
        #if axi is not None:
      ax.set_xlim(0,45);ax.set_ylim(-35,35);ax.set_zlim(0,15)

      ax.set_xlabel('X (m)')
      ax.set_ylabel('Y (m)')
      ax.set_zlabel('Z (m)')
      #ax.dist = 15
      ax.view_init(13,-178)
      ax2 = fig.add_subplot(212, projection='3d')
      ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
      ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
      ax2.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
      for ti in rt:

        for i in range(self.V.NumBeams):
              if ti==rt[0]:
                  x = self.BeamSeg[i].NodeX[:,0]
                  y = self.BeamSeg[i].NodeX[:,1]
                  z = self.BeamSeg[i].NodeX[:,2]
                  if i==0:
                      ax2.plot(x, y, z,c='k', marker='o',markersize=3,label='Initial Configuration')
                  else:
                      ax2.plot(x, y, z, c='k', marker='o',markersize=3)
              #rerr = self.ra0[i]-self.BeamSeg[i].NodeX
              rx= self.ra[i][:,ti,0]#-rerr[:,0]
              ry= self.ra[i][:,ti,1]#-rerr[:,1]
              rz= self.ra[i][:,ti,2]#-rerr[:,2]
              #ax2.scatter(x, y, z, c='r', marker='o')
              if i==0 and ti==rt[0]:
                  ax2.plot(rx,ry,rz,c='steelblue',linewidth=1.5,label='Intrinsic Theory')
              else:
                  ax2.plot(rx,ry,rz,c='steelblue',linewidth=1.5)

        # if self.results_nastran:
        #     for ni in nastran:
        #             figx = ax2.plot(self.rn[ni,:,0],self.rn[ni,:,1],self.rn[ni,:,2],'--')
      for ni in nastran:
            for i in range(self.V.NumBeams):
                if i in self.V.BeamsClamped:
                    #pdb.set_trace()
                    sx = np.hstack((self.V.ClampX[0],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],0]))
                    sy = np.hstack((self.V.ClampX[1],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],1]))
                    sz = np.hstack((self.V.ClampX[2],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],2]))

                else:
                    sx = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]].NodeOrder[-1],0],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],0]))
                    sy = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]].NodeOrder[-1],1],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],1]))
                    sz = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]].NodeOrder[-1],2],self.rn[ni,self.BeamSeg[i].NodeOrder[1:],2]))
                if i==0 and ni==nastran[0]:

                    figx = ax2.plot(sx,sy,sz,'k--',linewidth=1.5,label='Nastran Nonlinear')
                else:
                    figx = ax2.plot(sx,sy,sz,'k--',linewidth=1.5)

      #ax2.grid(False)
        #if ax2i is not None:
      ax2.set_xlim(0,45);ax2.set_ylim(-35,35);ax2.set_zlim(0,15)
      ax2.set_xlabel('X (m)')
      ax2.set_ylabel('Y (m)')
      ax2.set_zlabel('Z (m)')
      #ax2.dist = 5
      ax2.view_init(8,-100)
      plt.tight_layout()
      plt.legend(loc='upper left', bbox_to_anchor=(legend1[0],legend1[1]),
                 ncol=2, fancybox=True, shadow=True,fontsize=10)
      if save_static:
          plt.savefig(static_save,dpi=dpi_static)
      else:
          plt.show()


    def Animation3D(self,interval,Axes):

        # set up figure and animation
        fig = plt.figure()
        #ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
        #                     xlim=(-2, 2), ylim=(-2, 2))
        #ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(0, 27), ylim=(0, 27))
        ax = p3.Axes3D(fig)

        line, = ax.plot([],[],[], 'o', markersize=3)
        time_text = ax.text(0.5, 0.5,0.5, '', transform=ax.transAxes)
        #time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        #energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        time_text.set_text('')
        #def init():
        #    """initialize animation"""
        #    line.set_data([], [],[])
            #time_text.set_text('')
            #energy_text.set_text('')
        #    return line,# time_text, energy_text

        def animate(ti):
            """perform animation step"""
            #global pendulum, dt
            #pendulum.step(dt)

            #for i in range(len(self.ra)):

            line.set_data(np.hstack([self.ra[i][:,ti,0] for i in range(self.V.NumBeams)]),np.hstack([self.ra[i][:,ti,1] for i in range(self.V.NumBeams)]))
            line.set_3d_properties(np.hstack([self.ra[i][:,ti,2] for i in range(self.V.NumBeams)]))
            #time_text.set_text('time = %.1f' % pendulum.time_elapsed)
            #energy_text.set_text('energy = %.3f J' % pendulum.energy())
            tix = ti*self.V.dt
            time_text.set_text('time = %.1f' % tix)
            return line, time_text#, energy_text

        # Setting the axes properties
        ax.set_xlim3d(Axes[0])
        ax.set_xlabel('X')

        ax.set_ylim3d(Axes[1])
        ax.set_ylabel('Y')

        ax.set_zlim3d(Axes[2])
        ax.set_zlabel('Z')

        ax.set_title('3D Test')
        ax.view_init(9, -169)

        ani = animation.FuncAnimation(fig, animate, frames=self.V.tn,
                                      interval=interval,blit=True)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=45, metadata=dict(artist='Me') )#, bitrate=1800)
        ani.save('Figures/velocity.mp4', writer=writer)
        plt.show()

    def mode_disp(self,modes,phi,angular,Axes=[None,None,None]):


        for k in modes:
            fig=plt.figure()
            figx = fig.add_subplot(111, projection='3d')
            for i in range(self.V.NumBeams):

                x = self.BeamSeg[i].NodeX[0:,0]
                y = self.BeamSeg[i].NodeX[0:,1]
                z = self.BeamSeg[i].NodeX[0:,2]
                x2 = (self.BeamSeg[i].NodeX[0:-1,0]+self.BeamSeg[i].NodeX[1:,0])/2
                y2 = (self.BeamSeg[i].NodeX[0:-1,1]+self.BeamSeg[i].NodeX[1:,1])/2
                z2 = (self.BeamSeg[i].NodeX[0:-1,2]+self.BeamSeg[i].NodeX[1:,2])/2

                x0=x[0]*np.ones(len(x))
                y0=y[0]*np.ones(len(y))
                z0=z[0]*np.ones(len(z))

                figx.plot(x, y, z, c='grey', marker='o',markersize=0.5,lw=0.7)
                if phi==0:

                    scale=max([max([max(abs(self.Phi0[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    scale=scale*0.1
                    u=self.Phi0[i][k][:,3*angular+0]/scale
                    v=self.Phi0[i][k][:,3*angular+1]/scale
                    w=self.Phi0[i][k][:,3*angular+2]/scale
                    figx.plot(u+x,v+y,w+z,c='k',lw=2)
                if phi==3:

                    scale=max([max([max(abs(self.Phi0[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    scale=scale*0.1
                    u=self.Phi0[i][k][:,3*angular+0]/scale
                    v=self.Phi0[i][k][:,3*angular+1]/scale
                    w=self.Phi0[i][k][:,3*angular+2]/scale
                    figx.quiver(x, y, z, u, v, w,normalize=False)

                if phi==1:

                    scale=max([max([max(abs(self.Phi1[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    u=self.Phi1[i][k][:,3*angular+0]/scale
                    v=self.Phi1[i][k][:,3*angular+1]/scale
                    w=self.Phi1[i][k][:,3*angular+2]/scale
                    figx.plot(u+x,v+y,w+z,c='k',lw=2)

                if phi==2:

                    scale=max([max([max(abs(self.Phi2[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    scale=scale*0.1
                    u=self.Phi2[i][k][:-1,3*angular+0]/scale
                    v=self.Phi2[i][k][:-1,3*angular+1]/scale
                    w=self.Phi2[i][k][:-1,3*angular+2]/scale
                    figx.quiver(x2, y2, z2, u, v, w,normalize=False)

            if Axes[0] is not None:
                figx.set_xlim(Axes[0][0],Axes[0][1])
            if Axes[1] is not None:
                figx.set_ylim(Axes[1][0],Axes[1][1])
            if Axes[2] is not None:
                figx.set_zlim(Axes[2][0],Axes[2][1])

        plt.show()

    def mode_disp2(self,modes,phi,angular,Axes=[None,None,None]):

        kk=0
        kkn = len(modes)
        fig=plt.figure()
        for k in modes:
            kk=kk+1
            figx = fig.add_subplot(int((kkn+1)/2),2,kk, projection='3d')
            figx.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            figx.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            figx.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            for i in range(self.V.NumBeams):

                x = self.BeamSeg[i].NodeX[0:,0]
                y = self.BeamSeg[i].NodeX[0:,1]
                z = self.BeamSeg[i].NodeX[0:,2]
                x2 = (self.BeamSeg[i].NodeX[0:-1,0]+self.BeamSeg[i].NodeX[1:,0])/2
                y2 = (self.BeamSeg[i].NodeX[0:-1,1]+self.BeamSeg[i].NodeX[1:,1])/2
                z2 = (self.BeamSeg[i].NodeX[0:-1,2]+self.BeamSeg[i].NodeX[1:,2])/2

                x0=x[0]*np.ones(len(x))
                y0=y[0]*np.ones(len(y))
                z0=z[0]*np.ones(len(z))

                figx.plot(x, y, z, c='grey', marker='o',lw=0.7)
                if phi==0:

                    scale=max([max([max(abs(self.Phi0[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    scale=scale*0.1
                    u=self.Phi0[i][k][:,3*angular+0]/scale
                    v=self.Phi0[i][k][:,3*angular+1]/scale
                    w=self.Phi0[i][k][:,3*angular+2]/scale
                    figx.plot(u+x,v+y,w+z,c='k',lw=2)
                if phi==3:

                    scale=max([max([max(abs(self.Phi0[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    scale=scale*0.1
                    u=self.Phi0[i][k][:,3*angular+0]/scale
                    v=self.Phi0[i][k][:,3*angular+1]/scale
                    w=self.Phi0[i][k][:,3*angular+2]/scale
                    figx.quiver(x, y, z, u, v, w,normalize=False)

                if phi==1:

                    scale=max([max([max(abs(self.Phi1[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    u=self.Phi1[i][k][:,3*angular+0]/scale
                    v=self.Phi1[i][k][:,3*angular+1]/scale
                    w=self.Phi1[i][k][:,3*angular+2]/scale
                    figx.plot(u+x,v+y,w+z,c='k',lw=2)

                if phi==2:

                    scale=max([max([max(abs(self.Phi2[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    scale=scale*0.1
                    u=self.Phi2[i][k][:-1,3*angular+0]/scale
                    v=self.Phi2[i][k][:-1,3*angular+1]/scale
                    w=self.Phi2[i][k][:-1,3*angular+2]/scale
                    figx.quiver(x2, y2, z2, u, v, w,normalize=False)

            if Axes[0] is not None:
                figx.set_xlim(Axes[0][0],Axes[0][1])
            if Axes[1] is not None:
                figx.set_ylim(Axes[1][0],Axes[1][1])
            if Axes[2] is not None:
                figx.set_zlim(Axes[2][0],Axes[2][1])

        plt.tight_layout()
        plt.show()


    def mode_plot(self,modes,phi,angular,Axes=None):

        kk=0
        kkn = len(modes)
        fig=plt.figure(figsize=(9,3))
        for k in modes:
            kk=kk+1
            figx = fig.add_subplot(1,3,kk, projection='3d')
            figx.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            figx.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            figx.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            for i in range(self.V.NumBeams):

                x = self.BeamSeg[i].NodeX[0:,0]
                y = self.BeamSeg[i].NodeX[0:,1]
                z = self.BeamSeg[i].NodeX[0:,2]
                x2 = (self.BeamSeg[i].NodeX[0:-1,0]+self.BeamSeg[i].NodeX[1:,0])/2
                y2 = (self.BeamSeg[i].NodeX[0:-1,1]+self.BeamSeg[i].NodeX[1:,1])/2
                z2 = (self.BeamSeg[i].NodeX[0:-1,2]+self.BeamSeg[i].NodeX[1:,2])/2

                x0=x[0]*np.ones(len(x))
                y0=y[0]*np.ones(len(y))
                z0=z[0]*np.ones(len(z))

                figx.plot(x, y, z, c='k', marker='o',markersize=3,lw=1)

                if phi==1:

                    scale=max([max([max(abs(self.Phi1[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    scale=scale*0.1
                    if k==1:
                        u=-self.Phi1[i][k][:,3*angular+0]/scale
                        v=-self.Phi1[i][k][:,3*angular+1]/scale
                        w=-self.Phi1[i][k][:,3*angular+2]/scale
                    else:
                        u=self.Phi1[i][k][:,3*angular+0]/scale
                        v=self.Phi1[i][k][:,3*angular+1]/scale
                        w=self.Phi1[i][k][:,3*angular+2]/scale
                    figx.plot(u+x,v+y,w+z,c='steelblue',lw=2)
                    if Axes is not None:
                        figx.set_xlim3d(Axes[kk-1][0])
                        figx.set_ylim3d(Axes[kk-1][1])
                        figx.set_zlim3d(Axes[kk-1][2])
                    #figx.title.set_text('ff')
                    figx.set_xlabel('X (m)')
                    figx.set_ylabel('Y (m)')
                    figx.set_zlabel('Z (m)')
                    figx.view_init(view[kk-1][0],view[kk-1][1])

                if phi==2:

                    scale=max([max([max(abs(self.Phi2[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    scale=scale*0.1
                    if k==1:
                        u=-self.Phi2[i][k][:-1,3*angular+0]/scale
                        v=-self.Phi2[i][k][:-1,3*angular+1]/scale
                        w=-self.Phi2[i][k][:-1,3*angular+2]/scale
                    else:
                        u=self.Phi2[i][k][:-1,3*angular+0]/scale
                        v=self.Phi2[i][k][:-1,3*angular+1]/scale
                        w=self.Phi2[i][k][:-1,3*angular+2]/scale
                    figx.quiver(x2, y2, z2, u, v, w,color='steelblue',normalize=False)
                    if Axes is not None:
                        figx.set_xlim3d(Axes[kk-1][0])
                        figx.set_ylim3d(Axes[kk-1][1])
                        figx.set_zlim3d(Axes[kk-1][2])

                    figx.set_xlabel('X (m)')
                    figx.set_ylabel('Y (m)')
                    figx.set_zlabel('Z (m)')
                    figx.view_init(view[kk-1][0],view[kk-1][1])




        plt.tight_layout()
        if save_modes:
            plt.savefig(modes_save,dpi=300)
            #plt.close
        else:
            plt.show()


    def Rt(self,i,j,d,dis,nastran,external):

        fig = plt.figure()
        if dis:
            tna=np.shape(self.ra[0])[1]
            tia=np.linspace(0,self.V.tf,tna)
            ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
            figx = ax.plot(tia,self.ra[i][j,:,d]-self.ra0[i][j,d],label = 'Intrinsic Results')
            if self.results_nastran:
                    k=self.BeamSeg[i].NodeOrder[j]
                    tn=np.shape(self.un)[1]
                    ti=np.linspace(0,self.V.tf,tn)
                    figx = ax.plot(ti,self.un[0,:,k,d],'k--',label = 'Nastran Results')

        else:

            tna=np.shape(self.ra[0])[1]
            tia=np.linspace(0,self.V.tf,tna)
            ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
            #figx = ax.plot(tia,self.ra[i][j,:,d],label = 'BEAM = %s, POINT = %s, DIM = %s' % (i,j,d))
            figx = ax.plot(tia,self.ra[i][j,:,d],label = 'Intrinsic Results')

            if self.results_nastran:
                    k=self.BeamSeg[i].NodeOrder[j]
                    tn=np.shape(self.rn)[0]
                    ti=np.linspace(0,self.V.tf,tn)
                    figx = ax.plot(ti,self.rn[:,k,d],'--',label = 'Nastran Results')

        if self.externaldata:
          for di in external:
              if type(self.outData).__name__ == 'dict':
                  figx = ax.plot(np.asarray(self.outData[self.externaldata_name][di][0]),
                              np.asarray(self.outData[self.externaldata_name][di][1]),'x')
              if type(self.outData).__name__ == 'list':
                  figx = ax.plot(np.asarray(self.outData[di][0]),
                              np.asarray(self.outData[di][1]),'x')
        ax.set_xlabel('Time (sec.)')
        if d==0:
            ax.set_ylabel('Ux (m.)')
        if d==1:
            ax.set_ylabel('Uy (m.)')
        if d==2:
            ax.set_ylabel('Uz (m.)')

        plt.grid(linestyle='dotted')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


    def Rt2(self,i,j,dis,nastran,external):

        fig = plt.figure()
        for di in range(3):
            if dis:
                tna=np.shape(self.ra[0])[1]
                tia=np.linspace(0,self.V.tf,tna)
                ax = fig.add_subplot(eval('31%s'%(di+1)))
                figx = ax.plot(tia,self.ra[i][j,:,di]-self.ra0[i][j,di],label = 'Intrinsic Results')
                if self.results_nastran:
                        k=self.BeamSeg[i].NodeOrder[j]
                        tn=np.shape(self.un)[1]
                        ti=np.linspace(0,self.V.tf,tn)
                        figx = ax.plot(ti,self.un[0,:,k,di],'k--',label = 'Nastran Results')

            else:

                tna=np.shape(self.ra[0])[1]
                tia=np.linspace(0,self.V.tf,tna)
                ax = fig.add_subplot(eval('31%s'%(di+1)))
                #figx = ax.plot(tia,self.ra[i][j,:,d],label = 'BEAM = %s, POINT = %s, DIM = %s' % (i,j,d))
                figx = ax.plot(tia,self.ra[i][j,:,di],label = 'Intrinsic Results')

                if self.results_nastran:
                        k=self.BeamSeg[i].NodeOrder[j]
                        tn=np.shape(self.rn)[0]
                        ti=np.linspace(0,self.V.tf,tn)
                        figx = ax.plot(ti,self.rn[:,k,di],'--',label = 'Nastran Results')

            ax.set_xlabel('Time (sec.)')
            if di==0:
                ax.set_ylabel('Ux (m.)')
                ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
            if di==1:
                ax.set_ylabel('Uy (m.)')
            if di==2:
                ax.set_ylabel('Uz (m.)')

            plt.grid(linestyle='dotted')

        plt.show()


def qplot(model,rt,same_plot=0):

        fig = plt.figure(figsize=(7.2,5.4))

        for i in rt:
                ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
                if i==1:
                    figx = ax.plot(model.V.ti,model.q[:,i],c='steelblue',linewidth=1.2,label = 'Mode %s' % (i+1))
                elif i==3:
                    figx = ax.plot(model.V.ti,model.q[:,i],c='grey',linestyle='-',linewidth=1.2,label = 'Mode %s' % (i+1))
                elif i==5:
                    figx = ax.plot(model.V.ti,model.q[:,i],c='darkorange',linestyle='-',linewidth=1.2,label = 'Mode %s' % (i+1))
                elif i==7:
                    figx = ax.plot(model.V.ti,model.q[:,i],c='r',linestyle='-',linewidth=1.2,label = 'Mode %s' % (i+1))
                elif i==8:
                    figx = ax.plot(model.V.ti,model.q[:,i],'k',linewidth=1.2,label = 'Mode %s' % (i+1))
                elif i==9:
                    figx = ax.plot(model.V.ti,model.q[:,i],'k--',linewidth=1.2,label = 'Modes 1, 3, 5, 7, 10')

        #ax.set_ylabel('$q_1$',fontsize=14)
        ax.set_xlabel('Time (sec.)',fontsize=14)
        plt.xlim([0,5])
        ax.tick_params(labelsize='large' )
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #           ncol=3, mode="expand", borderaxespad=0.,fontsize=12)

        if save_q:
            fig.savefig(q_save,dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
