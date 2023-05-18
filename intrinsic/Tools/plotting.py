#========================================================================================================================================
 # Plotting
#=======================================================================================================================================

import matplotlib.pyplot as plt
from matplotlib import animation
import os
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import pdb
import importlib
import pickle
import Runs.Torun
import intrinsic.geometry

class PlotX:
    """Class to plot modes and solutions of a given model """

    def __init__(self,torun,results,results_modes,results_nastran=None,nastrandata=None,externaldata=None,externaldata_name=None):


        self.torun = torun
        self.V = importlib.import_module("Runs"+'.'+self.torun+'.'+'V')
        self.results = self.V.feminas_dir+self.V.model_name+'/' + results
        self.results_modes = self.V.feminas_dir+self.V.model_name+'/' + results_modes
        if results_nastran is not None:
            self.results_nastran = self.V.feminas_dir+self.V.model_name+'/'+results_nastran
            if 'r' in nastrandata.keys():
                self.rn = np.load(self.results_nastran+nastrandata['r'])
            if 'u' in nastrandata.keys():
                self.un = np.load(self.results_nastran+nastrandata['u'])
            try:
                self.tni = np.load(self.results_nastran+'tni.npy')
            except:
                tnn=np.shape(self.rn)[0]
                self.tni = np.linspace(0,self.V.tf,tnn)
        else:
            self.results_nastran = None


        self.externaldata = externaldata
        self.externaldata_name = externaldata_name
        self.BeamSeg, self.NumNode, self.NumNodes, self.DupNodes, self.inverseconn = intrinsic.geometry.geometry_def(self.V.Grid,self.V.NumBeams,self.V.BeamConn,self.V.start_reading,self.V.beam_start,self.V.nodeorder_start,self.V.node_start,self.V.Clamped,self.V.ClampX,self.V.BeamsClamped)

    def readData(self,local=0,sol='',q=''):

        self.nm='_'+str(self.V.NumModes)
        try:
            with open (self.results+'/Sol'+sol+'%s'%self.nm , 'rb') as fp:
                [self.ra0,self.ra,self.Rab,self.strain,self.kappa]  = pickle.load(fp)
        except:
            with open (self.results+'/Sol'+sol+'%s'%self.nm , 'rb') as fp:
                [self.ra0,self.ra,self.Rab]  = pickle.load(fp)

        try:
            self.tai = np.load(self.results+'ti'+'%s.npy'%self.nm)
        except:
            tna=np.shape(self.ra[0])[1]
            self.tai = np.linspace(0,self.V.tf,tna)
        if local:
            with open (self.results_modes+'/Phil%s'%self.nm , 'rb') as fp:
                self.Phi0,self.Phi1,self.Phi1m,self.Phi2,self.MPhi1,self.CPhi2x  = pickle.load(fp)
        else:
            with open (self.results_modes+'/Phi%s'%self.nm , 'rb') as fp:
                self.Phi0,self.Phi1,self.Phi1m,self.Phi2,self.MPhi1,self.CPhi2x  = pickle.load(fp)
        try:
            self.q = np.load(self.results+'/q'+q+'%s.npy'%self.nm)
        except:
            try:
              self.q = np.load(self.results+'/q2'+q+'%s.npy'%self.nm)
            except:
              print 'No Qs'
        if self.externaldata is not None:
            with open(self.V.feminas_dir+'/ExternalData/'+self.externaldata, 'rb') as handle:
                self.outData = pickle.load(handle)

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

                figx.plot(x, y, z, c='r', marker='o',lw=0.7)
                if phi==0:

                    scale=max([max([max(abs(self.Phi0[ix][k][jx,3*angular:3*angular+3])) for jx in range(self.BeamSeg[ix].EnumNodes)]) for ix in range(self.V.NumBeams)])
                    #scale=1
                    scale=scale*0.1
                    u=self.Phi0[i][k][:,3*angular+0]/scale
                    v=self.Phi0[i][k][:,3*angular+1]/scale
                    w=self.Phi0[i][k][:,3*angular+2]/scale
                    figx.plot(u+x,v+y,w+z,c='b',lw=2)
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
                    figx.plot(u+x,v+y,w+z,c='b',lw=2)

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


    def Disp3D0(self):

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for i in range(self.V.NumBeams):

          x = self.BeamSeg[i].NodeX[:,0]
          y = self.BeamSeg[i].NodeX[:,1]
          z = self.BeamSeg[i].NodeX[:,2]

          #ax.scatter(x, y, z, c='r', marker='o')
          ax.plot(x, y, z, c='r', marker='o')

      plt.show()


    def Static3DDisp(self,rt,nastran=[],axi=None):

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for ti in rt:

        for i in range(self.V.NumBeams):
              if ti==rt[0]:
                  x = self.BeamSeg[i].NodeX[:,0]
                  y = self.BeamSeg[i].NodeX[:,1]
                  z = self.BeamSeg[i].NodeX[:,2]
                  ax.plot(x, y, z, c='grey', marker='o')

              #pdb.set_trace()
              #rerr = self.ra0[i]-self.BeamSeg[i].NodeX
              rx= self.ra[i][:,ti,0]#-rerr[:,0]
              ry= self.ra[i][:,ti,1]#-rerr[:,1]
              rz= self.ra[i][:,ti,2]#-rerr[:,2]
              #ax.scatter(x, y, z, c='r', marker='o')
              ax.plot(rx,ry,rz,c='grey')

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
                figx = ax.plot(sx,sy,sz,'k--')

        ax.grid(False)
        if axi is not None:
            ax.set_xlim(axi[0],axi[1]);ax.set_ylim(axi[2],axi[3]);ax.set_zlim(axi[4],axi[5])
        fig.tight_layout()
      plt.show()

    def Dynamic3DDisp(self,rt,nastran=[],axi=None):

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for ti in rt:

        for i in range(self.V.NumBeams):


              rx= self.ra[i][:,ti,0]
              ry= self.ra[i][:,ti,1]
              rz= self.ra[i][:,ti,2]
              #ax.scatter(x, y, z, c='r', marker='o')
              ax.plot(rx,ry,rz,c='b')

        if self.results_nastran:
            for ni in nastran:
                    figx = ax.plot(self.rn[ni,:,0],self.rn[ni,:,1],self.rn[ni,:,2],'--')

        if axi is not None:
            ax.set_xlim(axi[0],axi[1]);ax.set_ylim(axi[2],axi[3]);ax.set_zlim(axi[4],axi[5])

      plt.show()


    def Dynamic2DDisp(self,dimen,rt,axi=None,external=None,nastran=[]):

        fig, ax = plt.subplots()
        for ti in rt:
            for i in range(len(self.ra)):

                figx = ax.plot(self.ra[i][:,ti,dimen[0]],self.ra[i][:,ti,dimen[1]])

        if self.results_nastran:
            for ni in nastran:
                for i in range(self.V.NumBeams):

                    figx = ax.plot(self.rn[ni,:,dimen[0]],self.rn[ni,:,dimen[1]],'--')

        if self.externaldata:
          for di in external:
              #pdb.set_trace()
              if type(self.outData).__name__ == 'dict':
                  figx = ax.plot(np.asarray(self.outData[self.externaldata_name][di][0]),
                              np.asarray(self.outData[self.externaldata_name][di][1]),'x'+marker[di+1])
              if type(self.outData).__name__ == 'list':
                  figx = ax.plot(np.asarray(self.outData[di][0]),
                                 np.asarray(self.outData[di][1]),'--')

        ax.legend(loc='upper right')


        # Hide grid lines
        # ax.grid(False)

        # # Hide axes ticks
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        plt.show()



    def Static2DDisp(self,dimen,rt,axi=None,external=None,nastran=[]):

        marker = ['k','r','b','y','g','m','c']
        marker = marker*len(rt)

        fig, ax = plt.subplots()
        j=0
        for ti in rt:

            for i in range(len(self.ra)):
                if ti==rt[0]:
                  figx = ax.plot(self.ra0[i][:,dimen[0]],self.ra0[i][:,dimen[1]],'k-o')
                  #pass

                figx = ax.plot(self.ra[i][:,ti,dimen[0]],self.ra[i][:,ti,dimen[1]],marker[j])

            j=j+1

        if self.results_nastran:
            for ni in nastran:
                for i in range(self.V.NumBeams):

                    figx = ax.plot(self.rn[ni,:,dimen[0]],self.rn[ni,:,dimen[1]],'--'+marker[j])


        if self.externaldata:
          for di in external:
              if type(self.externaldata).__name__ == 'dict':
                  figx = ax.plot(np.asarray(self.outData[self.externaldata_name][di][0]),
                              np.asarray(self.outData[self.externaldata_name][di][1]),'x'+marker[di+1])
              if type(self.externaldata).__name__ == 'list':
                  figx = ax.plot(np.asarray(self.outData[di][0]),
                              np.asarray(self.outData[di][1]),'x'+marker[di+1])
        ax.legend(loc='upper right')
        #plt.legend(['Initial Configuration','F = 3.7KN','F = 12.1KN','F = 17.5KN','F = 39.3KN','F = 61KN','F = 109.5KN','F = 120KN','Argyris'], loc='best')

        #ax.legend(['Initial Configuration','F = 0.2KN','F = 0.4KN','F = 0.6KN','F = 0.8KN','F = 1KN','F = 1.2KN','F = 1.4KN','F = 1.6KN','F = 2KN'])
        plt.show()


    def qplot(self,rt,same_plot=0):

        fig = plt.figure()

        if same_plot:

            ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
            for i in rt:
                figx = ax.plot(self.V.ti,self.q[:-1,i],label = 'Mode %s' % i)

        else:

            for i in rt:
                ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
                figx = ax.plot(self.V.ti,self.q[:-1,i],label = 'Mode %s' % i)

        plt.legend()
        plt.show()

    def Rt(self,i,j,d,dis,nastran,external):

        fig = plt.figure()
        if dis:
            tna=np.shape(self.ra[0])[1]
            tia=np.linspace(0,self.V.tf,tna)
            ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
            figx = ax.plot(tia,self.ra[i][j,:,d]-self.ra0[i][j,d],label = 'BEAM = %s, POINT = %s, DIM = %s' % (i,j,d))
            if self.results_nastran:
                    k=self.BeamSeg[i].NodeOrder[j]
                    tn=np.shape(self.rn)[0]
                    ti=np.linspace(0,self.V.tf,tn)
                    figx = ax.plot(ti,self.un[:,k,d],'--')

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
            ax.set_ylabel('X (m.)')
        if d==1:
            ax.set_ylabel('Y (m.)')
        if d==2:
            ax.set_ylabel('Z (m.)')

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

    def Rs(self,i,j,d,dis,nastran,external):

        fig = plt.figure()
        if dis:

            ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
            for di in d:
                figx = ax.plot(self.tai,self.ra[i][j,:,di]-self.ra0[i][j,di],label = 'BEAM = %s, POINT = %s, DIM = %s' % (i,j,di))
            if self.results_nastran:
                k=self.BeamSeg[i].NodeOrder[j]
                for dn in nastran:
                    figx = ax.plot(self.tni,self.un[:,k,dn],'--')

        else:

            ax = fig.add_subplot(111)#, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
            for di in d:
                figx = ax.plot(self.tai,self.ra[i][j,:,di],label = 'BEAM = %s, POINT = %s, DIM = %s' % (i,j,di))

            if self.results_nastran:
                k=self.BeamSeg[i].NodeOrder[j]
                for dn in nastran:
                    figx = ax.plot(self.tni,self.rn[:,k,dn],'--')

        if self.externaldata:
          for di in external:
              if type(self.outData).__name__ == 'dict':
                  figx = ax.plot(np.asarray(self.outData[self.externaldata_name][di][0]),
                              np.asarray(self.outData[self.externaldata_name][di][1]),'x')
              if type(self.outData).__name__ == 'list':
                  figx = ax.plot(np.asarray(self.outData[di][0]),
                              np.asarray(self.outData[di][1]),'x')

        plt.legend()
        plt.show()


    def Animation2D(self,dimen,interval,Axes):

        # set up figure and animation
        fig = plt.figure()
        #ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
        #                     xlim=(-2, 2), ylim=(-2, 2))
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=Axes[0], ylim=Axes[1])
        ax.grid()

        line, = ax.plot([], [], lw=1)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        #energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        def init():
            """initialize animation"""
            line.set_data([], [])
            time_text.set_text('')
            #energy_text.set_text('')
            return line,# time_text, energy_text

        def animate(ti):
            """perform animation step"""
            #global pendulum, dt
            #pendulum.step(dt)

            #for i in range(len(self.ra)):

            line.set_data(np.hstack([self.ra[i][:,ti,dimen[0]] for i in range(self.V.NumBeams)]),np.hstack([self.ra[i][:,ti,dimen[1]] for i in range(self.V.NumBeams)]))
            tix = ti*self.V.dt
            time_text.set_text('time = %.1f' % tix)
            #energy_text.set_text('energy = %.3f J' % pendulum.energy())
            return line, time_text#, energy_text



        ani = animation.FuncAnimation(fig, animate, frames=self.V.tn,
                                      interval=interval, blit=True, init_func=init)

        plt.show()


    def Animation3D(self,interval,Axes):

        # set up figure and animation
        fig = plt.figure()
        #ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
        #                     xlim=(-2, 2), ylim=(-2, 2))
        #ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(0, 27), ylim=(0, 27))
        ax = p3.Axes3D(fig)

        line, = ax.plot([],[],[], 'o-', lw=2)
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

        ani = animation.FuncAnimation(fig, animate, frames=self.V.tn,
                                      interval=interval,blit=True)

        plt.show()


if (__name__ == '__main__'):
    #pass
    #try1=PlotX('ArgyrisFrame_20','Test/Results_F','Test/Results_modes','ut.npy')
    #try2=PlotX('Hesse_25','Test/Results_F','Test/Results_modes')
    #try3=PlotX('RafaBeam_25','Results','Results_modes')
    #try4=PlotX('Hesse_25','Test/Results_F2','Test/Results_modes')
    #ebner=PlotX('','Test/Results_F','Test/Results_modes','Bean400/ut.npy')
    #ebner.readData()
    # hesse=PlotX('Hesse_25n','Test/Results_Fdead1','Test/Results_modes',externaldata ='hesse68.pickle')
    # hesse.readData()
    # hesse50=PlotX('Hesse_50','Test/Results_Fdead1','Test/Results_modes',externaldata ='hesse68.pickle')
    # hesse50.readData()
    sp=PlotX('SailPlaneCnew','Test/Results_FRES','Test/Results_modes','SP400/ut.npy')
    sp.readData()
    sp.Static3DDisp([0,1,2,3,4],nastran=[0,1,2,3,4])
    print 'Running Plotting'
    #hesse.Dynamic2DDisp([0,1],[0,int(6400./10*1),int(6400./10*2.5),int(640*4),int(640*6),int(640.*8),int(640.*10)],external=[0,1,2,3,4,5])
    #hesse50.Dynamic2DDisp([0,1],[0,int(4999./10*1),int(4999./10*2.5),int(499.*4),int(499.*6),int(499.*8),int(499.*10)],external=[0,1,2,3,4,5])
#hesse.Dynamic2DDisp([0,1],[0,int(1000*1),int(1000*2.5),int(1000*4),int(1000*6),int(1000*8),int(1000*10)],external=[0,1,2,3,4,5])



















































'''
def displt(rt):

  for i in range(NumBeams):

    x = BeamSeg[i].NodeX[0:,0]
    y = BeamSeg[i].NodeX[0:,1]
    z = BeamSeg[i].NodeX[0:,2]

    xn = (BeamSeg[i].NodeX[0:-1,0]+BeamSeg[i].NodeX[1:,0])/2
    yn = (BeamSeg[i].NodeX[0:-1,1]+BeamSeg[i].NodeX[1:,1])/2
    zn = (BeamSeg[i].NodeX[0:-1,2]+BeamSeg[i].NodeX[1:,2])/2
    #ax.scatter(x, y, z, c='r', marker='o')
    fig=plt.figure()
    figx = fig.add_subplot(111, projection='3d')
    figx.plot(x, y, z, c='r', marker='o',lw=0.7)
    u=rt[:,0]
    v=rt[:,1]
    w=rt[:,2]

    figx.plot(x+u,y+v,z+w,c='b',lw=2)
    figx.axis('equal')
    plt.show()



def displt31(rt,rt2,rt3):

  for i in range(NumBeams):

    x = BeamSeg[i].NodeX[0:,0]
    y = BeamSeg[i].NodeX[0:,1]
    z = BeamSeg[i].NodeX[0:,2]

    xn = (BeamSeg[i].NodeX[0:-1,0]+BeamSeg[i].NodeX[1:,0])/2
    yn = (BeamSeg[i].NodeX[0:-1,1]+BeamSeg[i].NodeX[1:,1])/2
    zn = (BeamSeg[i].NodeX[0:-1,2]+BeamSeg[i].NodeX[1:,2])/2
    #ax.scatter(x, y, z, c='r', marker='o')
    fig=plt.figure()
    figx = fig.add_subplot(111, projection='3d')
    figx.plot(x, y, z, c='r', marker='o',lw=0.7, label='Initial Configuration')
    figx.legend()
    u=rt[:,0]
    v=rt[:,1]
    w=rt[:,2]
    u2=rt2[:,0]
    v2=rt2[:,1]
    w2=rt2[:,2]
    u3=rt3[:,0]
    v3=rt3[:,1]
    w3=rt3[:,2]
    x0=x[0]*np.ones(len(x))
    y0=y[0]*np.ones(len(y))
    z0=z[0]*np.ones(len(z))


    figx.plot(u+x0,v+y0,w+z0,c='b',lw=2, label='PYFEM2NL')
    figx.legend()
    figx.plot(x+u2,y+v2,z+w2,c='g',lw=2, label='Nastran 400')
    figx.legend()
    figx.plot(x+u3,y+v3,z+w3,c='k',lw=2, label='Nastran 101')
    figx.legend()
    figx.axis('equal')
    plt.show()



pix=['Momement load = $1\pi$','Momement load =$5\pi$','Momement load = $10\pi$','Momement load = $15\pi$','Momement load = $20\pi$']
marker=['k-^','k-s','k-o','k-v','k-D']


rtn=[f1n,f5n,f10n,f15n,f20n]
rt=[f1rs,f5rs,f10rs,f15rs,f20rs]

fig, ax = plt.subplots()
for i in range(len(rt)):

 figx = ax.plot(rt[i][:,0],rt[i][:,1],marker[i],markersize=5, linewidth=1,label=pix[i])
 if i==4:
  figx = ax.plot(rtn[i][:,0]+BeamSeg[0].NodeX[0:,0],rtn[i][:,1]+BeamSeg[0].NodeX[0:,1], 'k--', linewidth=2,label='Nastran Solution')
 else:
  figx = ax.plot(rtn[i][:,0]+BeamSeg[0].NodeX[0:,0],rtn[i][:,1]+BeamSeg[0].NodeX[0:,1], 'k--', linewidth=2)

plt.xlim([-2.5,10.5]);plt.ylim([-1,8])
ax.legend(loc='upper right')

plt.show()


rtn=[f20n]
rt=[BeamSeg[0].solRa]

fig, ax = plt.subplots()
for i in range(len(rt)):

 figx = ax.plot(rt[i][:,0],rt[i][:,1],marker[i],markersize=5, linewidth=1,label=pix[i])
 if i==4:
  figx = ax.plot(rtn[i][:,0]+BeamSeg[0].NodeX[0:,0],rtn[i][:,1]+BeamSeg[0].NodeX[0:,1], 'k--', linewidth=2,label='Nastran Solution')
 else:
  figx = ax.plot(rtn[i][:,0]+BeamSeg[0].NodeX[0:,0],rtn[i][:,1]+BeamSeg[0].NodeX[0:,1], 'k--', linewidth=2)

plt.xlim([-2.5,10.5]);plt.ylim([-1,8])
ax.legend(loc='upper right')

plt.show()




kappam=np.zeros((NumModes,NumNodes,3))
strainm=np.zeros((NumModes,NumNodes,3))
for i in range(NumBeams):
    for j in range(BeamSeg[i].EnumNode-1):
      for m in range(NumModes):

        kappam[m][BeamSeg[i].NodeOrder[j]] = BeamSeg[i].GlobalAxes.dot(Mode[m].CPhi2x[BeamSeg[i].NodeOrder[j]][3:6])*q_static[m]
        strainm[m][BeamSeg[i].NodeOrder[j]] = BeamSeg[i].GlobalAxes.dot(Mode[m].CPhi2x[BeamSeg[i].NodeOrder[j]][0:3])*q_static[m]


vec=np.zeros((NumNodes,3))
for i in range(NumModes):
   vec = vec+kappam[i]

vecs=np.zeros((NumNodes,3))
for i in range(NumModes):
   vecs = vecs+strainm[i]

mbend=[0,2,4,6,10,12,14,18,20,22,26,28,32,34,38,40,44,46,50,52,54,58,60,64,66,69,70,72,75,76,77,78,79,80,81,82,83,84,85,86,89,90,91,92,94,96]

mrange=[1,5,10,15]


kap=[]
stra=[]
for j in mrange:

  vec=np.zeros((NumNodes,3))
  vecs=np.zeros((NumNodes,3))
  for i in range(mrange[j]):
    vec = vec+kappam[i]
    vecs = vecs+strainm[i]

  kap.append(vec)
  stra.append(vecs)


#=====================================================================
mrange=[1,5,10,15,20]
kappam=np.zeros((NumModes,NumNodes,3))
strainm=np.zeros((NumModes,NumNodes,3))
for i in range(NumBeams):
    for j in range(BeamSeg[i].EnumNode-1):
      for m in range(len(q_static2)):

        kappam[m][BeamSeg[i].NodeOrder[j]] = BeamSeg[i].GlobalAxes.dot(Mode[m].CPhi2x[BeamSeg[i].NodeOrder[j]][3:6])*q_static2[m]
        strainm[m][BeamSeg[i].NodeOrder[j]] = BeamSeg[i].GlobalAxes.dot(Mode[m].CPhi2x[BeamSeg[i].NodeOrder[j]][0:3])*q_static2[m]

kap=[]
stra=[]
for j in mrange:

  vec=np.zeros((NumNodes,3))
  vecs=np.zeros((NumNodes,3))
  for i in mbend[0:j]:
    vec = vec+kappam[i]
    vecs = vecs+strainm[i]

  kap.append(vec)
  stra.append(vecs)
#=========================================================================




#=====================================================================
q_static2=q_static
mrange=[1,3,5,7,10,14,18,23,30,40,50]
mrange=[1,2,3,4,5,6,8,10,12,14,15,16,18,22,28,35,45,50]
kappam=np.zeros((NumModes,NumNodes,3))
strainm=np.zeros((NumModes,NumNodes,3))
for i in range(NumBeams):
    for j in range(BeamSeg[i].EnumNode-1):
      for m in range(len(q_static2)):

        kappam[m][BeamSeg[i].NodeOrder[j]] = BeamSeg[i].GlobalAxes.dot(Mode[m].CPhi2x[BeamSeg[i].NodeOrder[j]][3:6])*q_static2[m]
        strainm[m][BeamSeg[i].NodeOrder[j]] = BeamSeg[i].GlobalAxes.dot(Mode[m].CPhi2x[BeamSeg[i].NodeOrder[j]][0:3])*q_static2[m]

kap=[]
stra=[]
for j in mrange:

  vec=np.zeros((NumNodes,3))
  vecs=np.zeros((NumNodes,3))
  for i in range(j):
    vec = vec+kappam[i]
    vecs = vecs+strainm[i]

  kap.append(vec)
  stra.append(vecs)
#=========================================================================
err=[]
for i in range(len(mrange)):
  err2=0.
  for j in range(NumNode):

    err2=err2+(0.2*np.pi-kap[i][BeamSeg[0].NodeOrder][:-1,2][j])**2

  err.append(np.sqrt(err2/NumNode))



fig, ax = plt.subplots()
figx = ax.plot(mrange,err)
plt.show()






font = {'family': 'serif',
        'weight': 'normal',
        'size': 15,
        }

#legen=[,,,,,, '30 Bending Modes', '46 Bending Modes']
marker=['k','k','k','k','k-.','b-x']
lw=[2,2,2,2,2,2.5]
fig, ax = plt.subplots()
X = 0.5*(BeamSeg[0].NodeX[0:24,0]+BeamSeg[0].NodeX[1:25,0])
for k in range(len(mrange)-2):

  Y = kap[k][BeamSeg[0].NodeOrder][0:24][:,2]
  figx = ax.plot(X,Y,marker[k],linewidth=lw[k])

k=4
Y = kap[k][BeamSeg[0].NodeOrder][0:24][:,2]
figx = ax.plot(X,Y,'k-o',markersize=4,linewidth=lw[k],label='30 Bending Modes')

k=5
Y = kap[k][BeamSeg[0].NodeOrder][0:24][:,2]
figx = ax.plot(X,Y,marker[k],linewidth=lw[k],label='Complete Set of Bending Modes (46)')

Y= 2*np.pi/10*np.ones(len(X))
figx = ax.plot(X,Y,'--r',linewidth=lw[k], label='$Analytical Sol = 0.2\pi$')
plt.xlim([0,10.5]);plt.ylim([-0.1,1.2])
#ax.legend(loc='lower left',fontsize=12)
ax.legend(loc='upper right',fontsize=11)
plt.tick_params(labelsize=12)
plt.xlabel(' X ', fontdict=font)
plt.ylabel(' $\kappa$', fontdict=font)
plt.show()


font = {'family': 'serif',
        'weight': 'normal',
        'size': 15,
        }

legen=['1 Bending Mode','5 Bending Modes', '10 Bending Modes', '20 Bending Modes', '30 Bending Modes', '46 Bending Modes']
marker=['k-s','k-d','k-^','k-o','k-.','b-x']
lw=[2,2,2,2,2,2.5]
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
X = 0.5*(BeamSeg[0].NodeX[0:24,0]+BeamSeg[0].NodeX[1:25,0])
for k in range(len(mrange)):


  Y = kap[k][BeamSeg[0].NodeOrder][0:24][:,2]

  figx = ax.plot(X,Y,marker[k],linewidth=lw[k],label=legen[k],markersize=4.5)

Y= 2*np.pi/10*np.ones(len(X))
figx = ax.plot(X,Y,'--r',linewidth=lw[k], label='$0.2\pi$')
plt.xlim([0,10.5]);plt.ylim([-0.2,1.])
#plt.xlim([-1,10]);plt.ylim([-0.3,1.2])
#ax.legend(loc='lower left',fontsize=12)
ax.legend(loc='lower left',fontsize=10)
plt.tick_params(labelsize=12)
plt.xlabel(' X ', fontdict=font)
plt.ylabel(' $\kappa$', fontdict=font)
plt.show()

l
#rt=np.zeros((NumNodes,3))
#for  i in range(NumNodes):
# rt[i,:]=BeamSeg[0].solTstrain_dyn[-1][i].dot(BeamSeg[0].solRstrain_dyn[-1][i])

#displt3(rt)




def disp_t(time,rt):




    #ax.scatter(x, y, z, c='r', marker='o')
    fig=plt.figure()
    figx = fig.add_subplot(111)
    figx.plot(time, rt)
    #figx.axis('equal')
    plt.show()


i=0
displt2(BeamSeg[i].solRa_dyn[50],BeamSeg[i].solRvelocity[50][BeamSeg[i].NodeOrder])


i=0
rt=BeamSeg[0].solRvelocity[:,10,2]
#rt=BeamSeg[0].solRstrain_dyn[:,10,2]
disp_t(time,rt)

i=0
rt=BeamSeg[i].solRstrain_dyn[-1]
#rt=BeamSeg[i].solRa_dyn[-1]
rt=BeamSeg[0].solRa
rt=BeamSeg[0].solRstrain
displt3(rt)


#rmode=Mode[5].Phi0[BeamSeg[0].NodeOrder]
#modeplt(rmode)


def modeplt2x(modeplot,r):

  nor=0
  count=0
  for i in range(NumBeams):
    for j in range(len(Mode[modeplot].Phi2[BeamSeg[i].NodeOrder])):
      nor_new=np.linalg.norm(BeamSeg[i].GlobalAxes.dot(Mode[modeplot].Phi2[BeamSeg[i].NodeOrder][j,r:r+3]))
      if nor_new>nor:
       nor=nor_new
       count=count+1
    if count==0:
      nor=1


  scale=1
  nor=nor*scale
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')


  for i in range(NumBeams):

    x = BeamSeg[i].NodeX[0:,0]
    y = BeamSeg[i].NodeX[0:,1]
    z = BeamSeg[i].NodeX[0:,2]

    xn = (BeamSeg[i].NodeX[0:-1,0]+BeamSeg[i].NodeX[1:,0])/2
    yn = (BeamSeg[i].NodeX[0:-1,1]+BeamSeg[i].NodeX[1:,1])/2
    zn = (BeamSeg[i].NodeX[0:-1,2]+BeamSeg[i].NodeX[1:,2])/2
    #ax.scatter(x, y, z, c='r', marker='o')
    ax.plot(x, y, z, c='r', marker='o',lw=0.7)
    uvw=BeamSeg[i].GlobalAxes.dot(Mode[modeplot].Phi2[BeamSeg[i].NodeOrder][:-1,r:r+3].T)/nor
    u=uvw[0,0:]
    v=uvw[1,0:]
    w=uvw[2,0:]

    ax.quiver(xn, yn, zn, u, v, w,normalize=False)
    #ax.plot(xn+u,yn+v,zn+w,c='b',lw=2)
'''
