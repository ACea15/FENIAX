import numpy as np
import PostProcessing.plotting3 as pl
import Utils.op2reader as o2
import pdb
import os
import matplotlib.pyplot as plt

fol='./Test/2DProblem/results_16-5-2019_15-32/'
p2d = pl.PlotX(geometryX=fol+'Results_modes/Geometry',PhiX=fol+'Results_modes/Phi_120',solX=fol+'Sols_120')
p2d.readData()

u2d = np.load('Test/ArgyrisFrame2d_ut.npy')


fol='./Test/3DProblem/results_16-5-2019_15-36/'
p3d = pl.PlotX(geometryX=fol+'Results_modes/Geometry',solX=fol+'Sols_120')
p3d.readData()

u3d = np.load('Test/ArgyrisFrame3d_ut.npy')
################################################################

def read_folder(pix,F,Findex,Fop2='NASTRAN/runs/XRF1-144trim_',Fop2_index='',solX='/dummy/Sols_50',qX='/dummy/q_50.npy',tiX='/dummy/ti_50.npy',geometryX='./Results_modes/Geometry',PhiX='./Results_modes/Phi_50',read_intrinsic=1,read_op2=1,rotate_op2=1):

    if read_intrinsic:
        for findex in Findex:
            pix[findex] = pl.PlotX(geometryX=geometryX,PhiX=PhiX,solX=os.getcwd()+solX.replace('dummy','%s%s'%(F,findex)),qX=os.getcwd()+qX.replace('dummy','%s%s'%(F,findex)),tiX=os.getcwd()+tiX.replace('dummy','%s%s'%(F,findex)))
            pix[findex].readData()

            
            nm = eval(qX.split('_')[-1].split('.')[0])
            pix[findex].elevator = np.load('./%s%s/q_elevator_%s.npy'%(F,findex,nm))
            pix[findex].u0=pix[findex].q[:,2*nm:3*nm].dot(pix[findex].Phi1[0][:,0])

            pix[findex].alpha = np.zeros(len(pix[findex].ra[0][0]))
            for ti in range(len(pix[findex].ra[0][0])):
                pix[findex].alpha[ti] = np.arccos(((pix[findex].ra[0][:,ti][1] - pix[findex].ra[0][:,ti][0]).dot(pix[findex].ra[0][:,0][1] - pix[findex].ra[0][:,0][0]))/(np.linalg.norm(pix[findex].ra[0][:,ti][1] - pix[findex].ra[0][:,ti][0])*np.linalg.norm(pix[findex].ra[0][:,0][1] - pix[findex].ra[0][:,0][0])))

    else:
        class Structure():
          pass

        pix[findex] = Structure()  

    if read_op2:

        for findex in range(len(Findex)):
            op2=o2.NastranReader('%s%s'%(Fop2,Fop2_index[findex]),'NASTRAN/XRF1-144trim',static=1)
            op2.readModel()
            t,r = op2.position()
            #op2.plot_asets()
            pix[Findex[findex]].rn = r


    if rotate_op2:
        for findex in Findex:
            pix[findex].rnn = pix[findex].rn[:]
            rrb = []
            for ix in range(len(pix[findex].rn[0])):

                for i in range(len(pix[findex].BeamSeg)):

                    if ix in pix[findex].BeamSeg[i]['NodeOrder']:
                        j = pix[findex].BeamSeg[i]['NodeOrder'].index(ix)

                        rrb.append(pix[findex].Rab[0][0,-1].dot(pix[findex].Rab[0][0,0].T).dot(pix[findex].BeamSeg[i]['NodeX'][j]-pix[findex].BeamSeg[0]['NodeX'][0]+pix[findex].rn[0,ix]-pix[findex].BeamSeg[i]['NodeX'][j])+pix[findex].BeamSeg[0]['NodeX'][0])

            pix[findex].rn=np.array([rrb])


#p1.Static3DDisp([-1],nastran=[0],BeamsClamped=[],axi=[0,70,-35,35,-10,40])
#p1.Static3DDisp([-1],BeamsClamped=[0],ClampX=[33.0221,0.,-1.48556],nastran=[0],axi=[0,70,-35,35,-5,20])



pix1 = {}
pix70 = {}

read_folder(pix1,'t',['211','221','231','241','251','261'],Fop2='./NASTRAN/runs/trim211/XRF1-144trim-',Fop2_index=['1g','2g','3g','3_5g','2_5g','1_5g'],solX='/dummy/Sols_50',qX='/dummy/q_50.npy',tiX='/dummy/ti_50.npy',geometryX='./Results_modes/Geometry',PhiX='./Results_modes/Phi_50',read_intrinsic=1,read_op2=1,rotate_op2=1)

read_folder(pix70,'t',['211','221','231','241','251','261'],Fop2='./NASTRAN/runs/trim211/XRF1-144trim-',Fop2_index=['1g','2g','3g','3_5g','2_5g','1_5g'],solX='/dummy/Sols_70',qX='/dummy/q_70.npy',tiX='/dummy/ti_70.npy',geometryX='./Results_modes/Geometry',PhiX='./Results_modes/Phi_70',read_intrinsic=1,read_op2=1,rotate_op2=1)


angle_nas = {}
angle_nas['141'] = 8
angle_nas['131'] = 4.095998E-01
angle_nas['121'] = 2.730666E-01
angle_nas['111'] = 1.365333E-01
angle_nas['261'] = 1.132856E-01
angle_nas['251'] = 1.888094E-01
angle_nas['241'] = 2.643331E-01
angle_nas['231'] = 2.265712E-01
angle_nas['221'] = 1.510475E-01
angle_nas['211'] = 7.552374E-02
angle_nas['341'] = 2.019865E-01
angle_nas['331'] = 1.514898E-01
angle_nas['321'] = 1.009932E-01
angle_nas['311'] = 5.049662E-02
angle_nas['441'] = 1.330938E-01
angle_nas['431'] = 9.982037E-02
angle_nas['421'] = 6.654691E-02
angle_nas['411'] = 3.327346E-02
angle_nas['541'] = 9.550321E-02
angle_nas['531'] = 7.162741E-02
angle_nas['521'] = 4.775160E-02
angle_nas['511'] = 2.387580E-02


elev_nas = {}
elev_nas['141'] = 3.892049E-01
elev_nas['131'] = 2.919037E-01
elev_nas['121'] = 1.946024E-01
elev_nas['111'] = 9.730122E-02
elev_nas['261'] = 6.947529E-02
elev_nas['251'] = 1.157921E-01
elev_nas['241'] = 1.621090E-01
elev_nas['231'] = 1.389506E-01
elev_nas['221'] = 9.263372E-02
elev_nas['211'] = 4.631686E-02
elev_nas['341'] = 1.013416E-01
elev_nas['331'] = 7.600618E-02
elev_nas['321'] = 5.067078E-02
elev_nas['311'] = 2.533539E-02
elev_nas['441'] = 4.060973E-02
elev_nas['431'] = 3.045730E-02
elev_nas['421'] = 2.030486E-02
elev_nas['411'] = 1.015243E-02
elev_nas['541'] = 1.126876E-03
elev_nas['531'] = 8.451568E-04
elev_nas['521'] = 5.634378E-04
elev_nas['511'] = 2.817189E-04


qdyn = [7076.0188,13313.525,20578.100,32401.25,4.652266E+04]
qdyn = [13313.525]
qdyni=[2]
loadings = [1.,1.5,2.,2.5,3.,3.5]
loadings_n = [1,6,2,5,3,4] 
#loadings = [1.,1.5,2.,2.5,3.,3.5]
#loadings_n = [1,6,2,5,3,4]
rad2deg=180./np.pi

# fig, ax = plt.subplots()
# for li in range(len(loadings)):
#     if li==0:
#         ax.plot(qdyn,[rad2deg*angle_nas['%s%s1' %(qdyni[i],loadings[li])] for i in range(len(qdyn))],'k',marker='s',linestyle='--',label='NASTRAN 144',fillstyle='none')
#         ax.plot(qdyn,[rad2deg*pix1['%s%s1'%(qdyni[i],loadings[li])].alpha[-1] for i in range(len(qdyn))],'b',marker='o',linestyle='none',label='NMROM',fillstyle='none')

#     else:
#         ax.plot(qdyn,[rad2deg*angle_nas['%s%s1' %(qdyni[i],loadings[li])] for i in range(len(qdyn))],'k',marker='s',linestyle='--',fillstyle='none')
#         ax.plot(qdyn,[rad2deg*pix1['%s%s1'%(qdyni[i],loadings[li])].alpha[-1] for i in range(len(qdyn))],'b',marker='o',linestyle='none',fillstyle='none')
#     #ax.plot(qdyn,[rad2deg*pix0['%s%s1'%(i+1,li+1)].alpha[-1] for i in range(len(qdyn))],'b',marker='^',linestyle='none',fillstyle='none')
#     #ax.plot(qdyn,[rad2deg*pix1['%s%s1'%(i+1,li+1)].u0[-1,4] for i in range(len(qdyn))],'b',marker='^',linestyle='none',fillstyle='none')
# ax.grid(True,linestyle='dotted')
# ax.legend()
# plt.xlabel('Dynamic pressure, $q_{\infty}$')
# plt.ylabel('Angle ($^o$)')
# plt.show()

def load2angle(pix1,loadings,loadings_n,qdyn,angle=1,elevator=0):
    if angle:
        fig, ax = plt.subplots()
        for i in range(len(qdyn)):
            if i==0:
                ax.plot(loadings,[rad2deg*angle_nas['%s%s1' %(qdyni[i],loadings_n[li])] for li in range(len(loadings_n))],'k',marker='s',linestyle='--',label='NASTRAN 144',fillstyle='none')
                ax.plot(loadings,[rad2deg*pix1['%s%s1'%(qdyni[i],loadings_n[li])].alpha[-1] for li in range(len(loadings_n))],'b',marker='o',linestyle='none',label='NMROM',fillstyle='none')

            else:
                ax.plot(loadings,[rad2deg*angle_nas['%s%s1' %(qdyni[i],loadings_n[li])] for li in range(len(loadings_n))],'k',marker='s',linestyle='--',label='NASTRAN 144',fillstyle='none')
                ax.plot(loadings,[rad2deg*pix1['%s%s1'%(qdyni[i],loadings_n[li])].alpha[-1] for li in range(len(loadings_n))],'b',marker='o',linestyle='none',label='NMROM',fillstyle='none')
            #ax.plot(qdyn,[rad2deg*pix0['%s%s1'%(i+1,li+1)].alpha[-1] for i in range(len(qdyn))],'b',marker='^',linestyle='none',fillstyle='none')
            #ax.plot(qdyn,[rad2deg*pix1['%s%s1'%(i+1,li+1)].u0[-1,4] for i in range(len(qdyn))],'b',marker='^',linestyle='none',fillstyle='none')
        ax.grid(True,linestyle='dotted')
        ax.legend()
        plt.xlabel('Loading (g)')
        plt.ylabel('Angle of attack($^o$)')
        plt.show()
    if elevator:
        fig, ax = plt.subplots()
        for i in range(len(qdyn)):
            if i==0:
                ax.plot(loadings,[rad2deg*elev_nas['%s%s1' %(qdyni[i],loadings_n[li])] for li in range(len(loadings_n))],'k',marker='s',linestyle='--',label='NASTRAN 144',fillstyle='none',markersize=8)
                ax.plot(loadings,[rad2deg*pix1['%s%s1'%(qdyni[i],loadings_n[li])].elevator[-1] for li in range(len(loadings_n))],'b',marker='o',linestyle='none',label='NMROM',fillstyle='none',markersize=8)

            else:
                ax.plot(loadings,[rad2deg*elev_nas['%s%s1' %(qdyni[i],loadings_n[li])] for li in range(len(loadings_n))],'k',marker='s',linestyle='--',label='NASTRAN 144',fillstyle='none',markersize=8)
                ax.plot(loadings,[rad2deg*pix1['%s%s1'%(qdyni[i],loadings_n[li])].elevator[-1] for li in range(len(loadings_n))],'b',marker='o',linestyle='none',label='NMROM',fillstyle='none',markersize=8)
            #ax.plot(qdyn,[rad2deg*pix0['%s%s1'%(i+1,li+1)].alpha[-1] for i in range(len(qdyn))],'b',marker='^',linestyle='none',fillstyle='none')
            #ax.plot(qdyn,[rad2deg*pix1['%s%s1'%(i+1,li+1)].u0[-1,4] for i in range(len(qdyn))],'b',marker='^',linestyle='none',fillstyle='none')
        ax.grid(True,linestyle='dotted')
        ax.legend()
        ax.set_ylim([0,12])
        ax.set_xlim([0,4])
        plt.xlabel('Loading (g)')
        plt.ylabel('Elevator angle ($^o$)')
        plt.show()

load2angle(pix1,loadings,loadings_n,qdyn,angle=1,elevator=1)

load2angle(pix70,loadings,loadings_n,qdyn,angle=1,elevator=1)

#################################################################################################

def X_time(pix1,m1,savefig=0):


    fig, ax = plt.subplots()
    ax.plot(pix1[m1].tai,rad2deg*pix1[m1].elevator[0::4],'b',linestyle='-')
    ax.grid(True,linestyle='dotted')
    ax.legend()
    plt.xlabel('Time (secs.)')
    plt.ylabel('Elevator deflection ($^o$)')
    if savefig:
        plt.savefig('./pics/fig_elevator%s.png'%m1)
    plt.show()


    fig, ax = plt.subplots()
    ax.plot(pix1[m1].tai,pix1[m1].ra[0][0,:][:,2]-pix1[m1].ra[0][0,0,2],'b',linestyle='-')
    ax.grid(True,linestyle='dotted')
    ax.legend()
    plt.xlabel('Time (secs.)')
    plt.ylabel('z-position (m)')
    if savefig:
        plt.savefig('./pics/Zg%s.png'%m1)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(pix1[m1].tai,rad2deg*pix1[m1].alpha,'b',linestyle='-')
    ax.grid(True,linestyle='dotted')
    ax.legend()
    plt.xlabel('Time (secs.)')
    plt.ylabel('alpha ($^o$)')
    if savefig:
        plt.savefig('./pics/alpha%s.png'%m1)
    plt.show()

X_time(pix1,m1='241',savefig=0)

X_time(pix70,m1='231',savefig=0)


p4 = {}

read_folder(p4,'t',['431'],Fop2='./NASTRAN/runs/trim211/XRF1-144trim-',Fop2_index=['3g'],solX='/dummy/Sols_50',qX='/dummy/q_50.npy',tiX='/dummy/ti_50.npy',geometryX='./Results_modes/Geometry',PhiX='./Results_modes/Phi_50',read_intrinsic=1,read_op2=1,rotate_op2=1)

X_time(p4,m1='431',savefig=0)


##################################################################################################


def Static3DDisp(model,rt,BeamsClamped=[0],ClampX=[0.,0.,0.],nastran=[],axi=None):

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for k in range(len(model)):
      for ti in rt:

        for i in range(model[k].NumBeams):
              if ti==rt[0]:
                  x = model[k].BeamSeg[i]['NodeX'][:,0]
                  y = model[k].BeamSeg[i]['NodeX'][:,1]
                  z = model[k].BeamSeg[i]['NodeX'][:,2]
                  ax.plot(x, y, z, c='grey', marker='o',markersize=3)

              #pdb.set_trace()
              #rerr = model[k].ra0[i]-model[k].BeamSeg[i]['NodeX']
              rx= model[k].ra[i][:,ti,0]#-rerr[:,0]
              ry= model[k].ra[i][:,ti,1]#-rerr[:,1]
              rz= model[k].ra[i][:,ti,2]#-rerr[:,2]
              #ax.scatter(x, y, z, c='r', marker='o')
              ax.plot(rx,ry,rz,c='grey')

        # if model[k].results_nastran:
        #     for ni in nastran:
        #             figx = ax.plot(model[k].rn[ni,:,0],model[k].rn[ni,:,1],model[k].rn[ni,:,2],'--')
        for ni in nastran:
            for i in range(model[k].NumBeams):
                if i in BeamsClamped:
                    #pdb.set_trace()
                    sx = np.hstack((ClampX[0],model[k].rn[ni,model[k].BeamSeg[i]['NodeOrder'][1:],0]))
                    sy = np.hstack((ClampX[1],model[k].rn[ni,model[k].BeamSeg[i]['NodeOrder'][1:],1]))
                    sz = np.hstack((ClampX[2],model[k].rn[ni,model[k].BeamSeg[i]['NodeOrder'][1:],2]))
                elif i==0 and (not BeamsClamped):
                    sx = []
                    sy = []
                    sz = []

                else:
                    sx = np.hstack((model[k].rn[ni,model[k].BeamSeg[model[k].inverseconn[i]]['NodeOrder'][-1],0],model[k].rn[ni,model[k].BeamSeg[i]['NodeOrder'][1:],0]))
                    sy = np.hstack((model[k].rn[ni,model[k].BeamSeg[model[k].inverseconn[i]]['NodeOrder'][-1],1],model[k].rn[ni,model[k].BeamSeg[i]['NodeOrder'][1:],1]))
                    sz = np.hstack((model[k].rn[ni,model[k].BeamSeg[model[k].inverseconn[i]]['NodeOrder'][-1],2],model[k].rn[ni,model[k].BeamSeg[i]['NodeOrder'][1:],2]))
                figx = ax.plot(sx,sy,sz,'k--')

  ax.grid(False)
  if axi is not None:
     ax.set_xlim(axi[0],axi[1]);ax.set_ylim(axi[2],axi[3]);ax.set_zlim(axi[4],axi[5])
  fig.tight_layout()
  plt.show()


Static3DDisp([pix1['211'],pix1['221'],pix1['241']],[-1],nastran=[0],BeamsClamped=[],axi=[0,70,-35,35,-10,15])

Static3DDisp([pix1['241']],[-1],nastran=[0],BeamsClamped=[],axi=[0,70,-35,35,-20,25])




class plot_various(pl.PlotX):

    def __init__(self,pi,geometryX=None,PhiX=None,solX=None,qX=None,tiX=None,nastranX=None,externalX=None):
        pl.PlotX.__init__(self,geometryX,PhiX,solX,qX,tiX,nastranX,externalX)
        self.pi = pi

    def Static3DDisp(self,rt,BeamsClamped=[0],ClampX=[0.,0.,0.],nastran=[],axi=None,pi=[]):

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for ti in rt:

        for i in range(self.NumBeams):
              if ti==rt[0]:
                  x = self.BeamSeg[i]['NodeX'][:,0]
                  y = self.BeamSeg[i]['NodeX'][:,1]
                  z = self.BeamSeg[i]['NodeX'][:,2]
                  ax.plot(x, y, z, c='grey', marker='o',markersize=3)

              #pdb.set_trace()
              #rerr = self.ra0[i]-self.BeamSeg[i]['NodeX']
              rx= self.ra[i][:,ti,0]#-rerr[:,0]
              ry= self.ra[i][:,ti,1]#-rerr[:,1]
              rz= self.ra[i][:,ti,2]#-rerr[:,2]
              #ax.scatter(x, y, z, c='r', marker='o')
              ax.plot(rx,ry,rz,c='grey')
              for xi in range(len(self.pi)):
                  rx= self.pi[xi].ra[i][:,ti,0]#-rerr[:,0]
                  ry= self.pi[xi].ra[i][:,ti,1]#-rerr[:,1]
                  rz= self.pi[xi].ra[i][:,ti,2]#-rerr[:,2]
                  #ax.scatter(x, y, z, c='r', marker='o')
                  ax.plot(rx,ry,rz,c='grey')
              
        # if self.results_nastran:
        #     for ni in nastran:
        #             figx = ax.plot(self.rn[ni,:,0],self.rn[ni,:,1],self.rn[ni,:,2],'--')
        for ni in nastran:
            for i in range(self.NumBeams):
                if i in BeamsClamped:
                    #pdb.set_trace()
                    sx = np.hstack((ClampX[0],self.rn[ni,self.BeamSeg[i]['NodeOrder'][1:],0]))
                    sy = np.hstack((ClampX[1],self.rn[ni,self.BeamSeg[i]['NodeOrder'][1:],1]))
                    sz = np.hstack((ClampX[2],self.rn[ni,self.BeamSeg[i]['NodeOrder'][1:],2]))

                else:
                    sx = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]]['NodeOrder'][-1],0],self.rn[ni,self.BeamSeg[i]['NodeOrder'][1:],0]))
                    sy = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]]['NodeOrder'][-1],1],self.rn[ni,self.BeamSeg[i]['NodeOrder'][1:],1]))
                    sz = np.hstack((self.rn[ni,self.BeamSeg[self.inverseconn[i]]['NodeOrder'][-1],2],self.rn[ni,self.BeamSeg[i]['NodeOrder'][1:],2]))
                figx = ax.plot(sx,sy,sz,'k--')

                for xi in range(len(self.pi)):
                    if i in BeamsClamped:
                        #pdb.set_trace()
                        sx = np.hstack((ClampX[0],self.pi[xi].rn[ni,self.BeamSeg[i]['NodeOrder'][1:],0]))
                        sy = np.hstack((ClampX[1],self.pi[xi].rn[ni,self.BeamSeg[i]['NodeOrder'][1:],1]))
                        sz = np.hstack((ClampX[2],self.pi[xi].rn[ni,self.BeamSeg[i]['NodeOrder'][1:],2]))

                    else:
                        sx = np.hstack((self.pi[xi].rn[ni,self.BeamSeg[self.inverseconn[i]]['NodeOrder'][-1],0],self.pi[xi].rn[ni,self.BeamSeg[i]['NodeOrder'][1:],0]))
                        sy = np.hstack((self.pi[xi].rn[ni,self.BeamSeg[self.inverseconn[i]]['NodeOrder'][-1],1],self.pi[xi].rn[ni,self.BeamSeg[i]['NodeOrder'][1:],1]))
                        sz = np.hstack((self.pi[xi].rn[ni,self.BeamSeg[self.inverseconn[i]]['NodeOrder'][-1],2],self.pi[xi].rn[ni,self.BeamSeg[i]['NodeOrder'][1:],2]))
                    figx = ax.plot(sx,sy,sz,'k--')

        ax.grid(False)
        if axi is not None:
            ax.set_xlim(axi[0],axi[1]);ax.set_ylim(axi[2],axi[3]);ax.set_zlim(axi[4],axi[5])
        fig.tight_layout()
      plt.show()
