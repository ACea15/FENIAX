from scipy.interpolate import interp1d
import numpy as np
import pdb
import importlib
import multiprocessing
import time
from Tools.transformations import quaternion_rotation6,quaternion_conjugate
from intrinsic.functions import tilde,H0,H1

import Runs.Torun
#Runs.Torun.torun = 'ArgyrisFrame_20'
#V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+'V')
V = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.variables)
from intrinsic.functions import Matrix_rotation6
#import intrinsic.geometry
#BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = intrinsic.geometry.geometry_def(
#V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,
#V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)

class Force:
    """Class to define any type of external force."""


    def __init__(self,Phi1,Gravity=None,Phig0=None,BeamSeg=None,NumFLoads=0,NumDLoads=0,NumALoads=0,
                 Follower_points_app=None,Follower_interpol=None,
                 Dead_points_app=None,Dead_interpol=None):

        self.Phi1 = Phi1
        self.Gravity = Gravity
        self.Phig0 = Phig0
        self.BeamSeg = BeamSeg
        # self.NumSLoads = NumSLoads
        # if NumSLoads>0:

        #     if len(Static_points_app)!=NumSLoads:
        #        raise ValueError('Number of point loads not equal to application points')
        #     else:
        #        self.Static_points_app = Static_points_app
        #     if len(Static_interpol)!=NumSLoads:
        #        raise ValueError('Number of interpolation array not equal to Number of loads')
        #     else:
        #        self.Static_interpol = Static_interpol
        #     self.NumSDimen = [len(Static_points_app[i][2]) for i in range(NumSLoads)]
        self.NumDLoads = NumDLoads
        if NumDLoads>0:

            if len(Dead_points_app)!=NumDLoads:
               raise ValueError('Number of point loads not equal to application points')
            else:
               self.Dead_points_app = Dead_points_app
            if len(Dead_interpol)!=NumDLoads:
               raise ValueError('Number of interpolation array not equal to Number of loads')
            else:
               self.Dead_interpol = Dead_interpol

            self.NumDDimen = [len(Dead_points_app[i][2]) for i in range(NumDLoads)]

        self.NumFLoads = NumFLoads
        if NumFLoads>0:

            if len(Follower_points_app)!=NumFLoads:
               raise ValueError('Number of point loads not equal to application points')
            else:
               self.Follower_points_app = Follower_points_app
            if len(Follower_interpol)!=NumFLoads:
               raise ValueError('Number of interpolation array not equal to Number of loads')
            else:
               self.Follower_interpol = Follower_interpol

            self.NumFDimen = [len(Follower_points_app[i][2]) for i in range(NumFLoads)]

        self.NumALoads = NumALoads
        if NumALoads>0:
            self.A = importlib.import_module("Runs"+'.'+Runs.Torun.torun+'.'+Runs.Torun.aero)
            self.AICsQhh = np.load(self.A.Amatrix)
            if self.A.rbdx:
                self.AICsQx = np.load(self.A.Axmatrix)
            if self.A.TrimOn:
                self.AICsQx = np.load(self.A.Axmatrix)
                self.q_elevator = [0.]
                self.force_spring = [0.]
                self.force_spring_dot = [0.]
                self.force_spring_int = [0.]
                self.force_damper = [0.]
                self.force_damper_dot = [0.]
                self.force_damper_int = [0.]
                self.tm1 = 0.
                self.rai = 0.
                self.Rabi = self.BeamSeg[0].GlobalAxes
                self.velocity_local = 0.
                self.omega_local = 0.
                self.q_elev0 = 0.
            if self.A.GustOn:
                self.AICsQhj = np.load(self.A.Agmatrix)               # Aerodynamic matrices for gusts after RFA
                self.npanels = np.shape(self.AICsQhj[0])[1]           # Number of aerodynamic panels
                self.Control_nodes = np.load(self.A.Control_nodes)    # Control nodes coordinates
                self.Panels_LE = np.load(self.A.Panels_LE)            # Min X-position of leading edge panel
                self.Panels_TE = np.load(self.A.Panels_TE)            # Max X-position of trailing edge panel
                self.Dihedral = np.load(self.A.Dihedral)              # Dihedral of each panel
                if self.A.Gust_shape_span == 'antisym_span_quad':
                    self.Gust_shape = (lambda y:y**2/self.A.wing_span**2)
                elif self.A.Gust_shape_span == 'antisym_span_lin':
                    self.Gust_shape =  (lambda y:y/self.A.wing_span)
                elif self.A.Gust_shape_span == 'darpa_span':
                    self.Gust_shape =  (lambda y:np.cos(np.pi*(self.A.wing_span-y)/self.A.wing_span))
                else:
                    self.Gust_shape = (lambda y:1.)

    def interpolation(self):

        # if self.NumSLoads>0:
        #     self.Staticinterpol = [[[] for d in range(self.NumSDimen[i]) ] for i in range(self.NumSLoads)]
        #     for i in range(self.NumSLoads):
        #       for d in range(self.NumSDimen[i]):
        #         self.Staticinterpol[i][d] = interp1d(self.Static_interpol[i][d][0],self.Static_interpol[i][d][1])

        if self.NumFLoads>0:
            self.Followerinterpol = [[[] for d in range(self.NumFDimen[i]) ] for i in range(self.NumFLoads)]
            for i in range(self.NumFLoads):
              for d in range(self.NumFDimen[i]):
                self.Followerinterpol[i][d] = interp1d(self.Follower_interpol[i][d][0],self.Follower_interpol[i][d][1])

        if self.NumDLoads>0:
            self.Deadinterpol = [[[] for d in range(self.NumDDimen[i]) ] for i in range(self.NumDLoads)]
            for i in range(self.NumDLoads):
              for d in range(self.NumDDimen[i]):
                self.Deadinterpol[i][d] = interp1d(self.Dead_interpol[i][d][0],self.Dead_interpol[i][d][1])

    def interpolation_gust(self):
        """Function to generate the interpolation of the gust loads Fgust_tot_time_interp[mode](time), Fgust_lag_time_interp[lag][mode](time)"""

        if self.A.Gust_type.lower()=='1mc':
            coeff2=2.*np.pi*self.A.u_inf/self.A.L_g
        elif self.A.Gust_type.upper()=='RNDM':
            pass
            #t_max_gust=max(Gust_RNDM[:,0])
            #time_gust = Gust_RNDM[:,0]
            #ntime_gust=len(Gust_RNDM[:,0])

        ntime_gust = len(self.A.time_gust)
        #Gust matrices definition
        Gust=np.zeros((self.npanels))                 #Downwash,Downwash_dot and Downwash_dot_dot
        Gust_dot=np.zeros((self.npanels))
        Gust_ddot=np.zeros((self.npanels))
        Gust_old=np.zeros((self.npanels))
        Gust_dot_old=np.zeros((self.npanels))

        #Flight and gust data
        #q_dyn=self.A.u_inf*self.A.u_inf*self.A.rho_inf*0.5
        coeff1=self.A.c/(2.*self.A.u_inf)

        #Define Gust starting and finishing time
        if self.A.Gust_type.lower() =='1mc':
            Time_finish_gust =self.A.Time_start_gust+self.A.L_g/self.A.u_inf
        elif self.A.Gust_type.upper() =='RNDM':
            Time_finish_gust =self.A.Time_start_gust+t_max_gust
        delay_LE=(self.Panels_LE-self.A.X0_g)/self.A.u_inf           # Time delay at fist leading edge panel
        delay_TE=(self.Panels_TE-self.A.X0_g)/self.A.u_inf           # Time delay at last trailing edge panel

        #Initialize Gust forces
        Fgust_tot_time=np.zeros((V.NumModes,ntime_gust))

        #Initialize Gust lag forces
        Fgust_lag_time=[]
        for lag in range (self.A.NumPoles):
            Fgust_lag_time.append(np.zeros((V.NumModes,ntime_gust)))

        ##############################################################
        #gust profile definition for each time step
        ##############################################################

        counter_time=0
        #Evaluate an iterpolation function for RNDM gust profile
        if self.A.Gust_type.upper()=='RNDM':
            Gust_RNDM_time=interp1d(Gust_RNDM[:,0],Gust_RNDM[:,1],kind='linear')

        for time_ in self.A.time_gust:
            if (time_>=self.A.Time_start_gust+delay_LE and time_<=Time_finish_gust+delay_TE):
                for panel in range (self.npanels):
                    delay=(self.Control_nodes[panel,0]-self.A.X0_g)/self.A.u_inf
                    shape_span = self.Gust_shape(self.Control_nodes[panel,1])
                    if (time_>=self.A.Time_start_gust+delay and time_<=Time_finish_gust+delay):
                        if self.A.Gust_type.lower()=='1mc':
                            Gust[panel]=shape_span*self.Dihedral[panel]*(self.A.V0_g/(self.A.u_inf*2))*(1-np.cos(coeff2*(time_-self.A.Time_start_gust-delay)))
                            Gust_dot[panel]=shape_span*self.Dihedral[panel]*(self.A.V0_g/(self.A.u_inf*2))*np.sin(coeff2*(time_-self.A.Time_start_gust-delay))*coeff2
                            Gust_ddot[panel]=shape_span*self.Dihedral[panel]*(self.A.V0_g/(self.A.u_inf*2))*np.cos(coeff2*(time_-self.A.Time_start_gust-delay))*coeff2**2
                        elif self.A.Gust_type.upper()=='RNDM':
                            Gust[panel]=self.Dihedral[panel]*np.arctan(Gust_RNDM_time(time_-self.A.Time_start_gust-delay)/self.A.u_inf)
                            #Gust_dot[panel]=0.
                            #Gust_ddot[panel]=0.
                            if counter_time==0:
                                Gust_dot[panel]=0.
                                Gust_ddot[panel]=0.
                            if counter_time>0:
                                Gust_dot[panel]=(Gust[panel]-Gust_old[panel])/(self.A.time_gust[counter_time]-self.A.time_gust[counter_time-1])
                                Gust_old[panel]=Gust[panel]
                            if counter_time>1:
                                 Gust_ddot[panel]=(Gust_dot[panel]-Gust_dot_old[panel])/(self.A.time_gust[counter_time]-self.A.time_gust[counter_time-1])
                                 Gust_dot_old[panel]=Gust_dot[panel]

                    else:
                        Gust[panel]=0.
                        Gust_dot[panel]=0.
                        Gust_ddot[panel]=0.
                #Gust forces definition and storing
                Fgust = self.A.q_inf*np.dot(self.AICsQhj[0],Gust)
                Fgust_dot = self.A.q_inf*coeff1*np.dot(self.AICsQhj[1],Gust_dot)
                Fgust_ddot = self.A.q_inf*(coeff1**2)*np.dot(self.AICsQhj[2],Gust_ddot)
                Fgust_tot=Fgust+Fgust_dot+Fgust_ddot

            else:
                #Gust[:]=0.
                #Gust_dot[:]=0.
                #Gust_ddot[:]=0.

                #Gust forces definition and storing
                Fgust=np.zeros(V.NumModes)#q_dyn*np.dot(self.AICsQhj[0],Gust)
                Fgust_dot=np.zeros(V.NumModes)#q_dyn*coeff1*np.dot(self.AICsQhj[1],Gust_dot)
                Fgust_ddot=np.zeros(V.NumModes)#q_dyn*(coeff1**2)*np.dot(self.AICsQhj[2],Gust_ddot)
                Fgust_tot=np.zeros(V.NumModes)#Fgust+Fgust_dot+Fgust_ddot

            #Save gust loads time history
            Fgust_tot_time[:,counter_time]=Fgust_tot[:]
            for lag in range (self.A.NumPoles):
                Fgust_lag_time[lag][:,counter_time]=np.dot(self.AICsQhj[lag+3],Gust_dot)
            counter_time=counter_time+1

        #Generate an interpolation function providing the gust loads for a given time
        self.Fgust_tot_time_interp=[]
        for mode in range (V.NumModes):
            self.Fgust_tot_time_interp.append(interp1d(self.A.time_gust,Fgust_tot_time[mode,:],kind='linear'))

        self.Fgust_lag_time_interp=[[] for pix in range(self.A.NumPoles)]
        for lag in range (self.A.NumPoles):
            for mode in range (V.NumModes):
                self.Fgust_lag_time_interp[lag].append(interp1d(self.A.time_gust,Fgust_lag_time[lag][mode,:],kind='linear'))


    def follower_interpol(self,ix,t):
        #pdb.set_trace()
        if 'Followerinterpol' not in dir(self):
            self.interpolation()

        F = [0. for i in range(6)]
        for d in range(len(self.Follower_points_app[ix][2])):
            F[self.Follower_points_app[ix][2][d]] = self.Followerinterpol[ix][d](t)

        return np.asarray(F)

    def dead_interpol(self,ix,t):

        if 'Deadinterpol' not in dir(self):
            self.interpolation()

        F = [0. for i in range(6)]
        for d in range(len(self.Dead_points_app[ix][2])):
            F[self.Dead_points_app[ix][2][d]] = self.Deadinterpol[ix][d](t)

        return np.asarray(F)

    def gust_interpol(self,t):

        if 'Fgust_tot_time_interp' not in dir(self):
            self.interpolation_gust()

        F = np.zeros(V.NumModes)
        for mi in range(V.NumModes):
            F[mi] = self.Fgust_tot_time_interp[mi](t)
        return F

    def gust_lags_interpol(self,t):

        if 'Fgust_lag_time_interp' not in dir(self):
            self.interpolation_gust()

        F = np.zeros((self.A.NumPoles,V.NumModes))
        for li in range(self.A.NumPoles):
            for mi in range(V.NumModes):
                F[li][mi] = self.Fgust_lag_time_interp[li][mi](t)
        return F


    def force_follower(self,t):
        F=[[] for i in range(self.NumFLoads)]
        for i in range(self.NumFLoads):
            F[i] = self.follower_interpol(i,t)
        return F

    def force_dead(self,t,rotation):
        #pdb.set_trace()
        F = [[] for i in range(self.NumDLoads)]
        for i in range(self.NumDLoads):
            F[i] = self.dead_interpol(i,t)

        Fr=[[] for i in range(self.NumDLoads)]
        #print np.shape(rotation)
        if np.shape(rotation) == (self.NumDLoads,4):
            for i in range(self.NumDLoads):
                Fr[i] = quaternion_rotation6(F[i],quaternion_conjugate(rotation[i]))
                #Fr[i] = quaternion_rotation6(F[i],rotation[i])
        elif np.shape(rotation) == (self.NumDLoads,3,3):
            for i in range(self.NumDLoads):
                Fr[i] = Matrix_rotation6(F[i],rotation[i].T)
                #Fr[i] = Matrix_rotation6(F[i],rotation[i])
        elif np.shape(rotation)[2] == 3:
            for i in range(self.NumDLoads):
                Fr[i] = Matrix_rotation6(F[i],rotation[self.Dead_points_app[i][0]][self.Dead_points_app[i][1]].T)
                #Fr[i] = Matrix_rotation6(F[i],rotation[i])

        return Fr

    def forceFollower_eta(self,t):

        F = self.force_follower(t)
        #print F,t
        eta = np.zeros(V.NumModes)
        for k in range(V.NumModes):
          for i in range(self.NumFLoads):

            eta[k] = eta[k]+ self.Phi1[self.Follower_points_app[i][0]][k][self.Follower_points_app[i][1]].dot(F[i])
            #eta[k] =  self.Phi1[self.Follower_points_app[i][0]][k][self.Follower_points_app[i][1]].dot(F[i])
        return eta

    def forceDead_eta(self,t,rotation):

        F = self.force_dead(t,rotation)
        eta = np.zeros(V.NumModes)
        for k in range(V.NumModes):
          for i in range(self.NumDLoads):
            eta[k] = eta[k]+self.Phi1[self.Dead_points_app[i][0]][k][self.Dead_points_app[i][1]].dot(F[i])
        return eta

    def forceGravity_eta(self,t,rotation):
        #pdb.set_trace()
        eta = np.zeros(V.NumModes-V.NumModes_res)
        for k in range(V.NumModes-V.NumModes_res):
          for i in range(V.NumBeams):
              for j in range(self.BeamSeg[i].EnumNodes):
                  if rotation:
                      eta[k] += self.Phi1[i][k][j].dot(Matrix_rotation6(self.Phig0[i][j],rotation[i][j].T))
                  else:
                      eta[k] += self.Phi1[i][k][j].dot(Matrix_rotation6(self.Phig0[i][j],self.BeamSeg[i].GlobalAxes.T))
        return eta


    def forceAero_eta(self,q):
        """Aerodynamic modal forces from modal exitacions"""

        eta = 0.5*self.A.rho_inf*(self.A.u_inf**2*self.AICsQhh[0,:,:].dot(q[2*V.NumModes:3*V.NumModes]) +
                                  self.A.u_inf*self.A.c/2*self.AICsQhh[1,:,:].dot(q[:V.NumModes]) +
                                  (self.A.c/2)**2*self.AICsQhh[2,:,:].dot(q[1*V.NumModes:2*V.NumModes]))
        for i in range(self.A.NumPoles):
            eta += 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICsQhh[i+3,:,:].dot(q[(i+3)*V.NumModes:(i+4)*V.NumModes])

        return eta

    def forceAero_etax(self,q,t=1.):
        """Aerodynamic modal force from modal exitacions, removing q1_dot component for construction of aeroelastic system"""

        eta = self.A.q_inf*self.AICsQhh[0,:,:].dot(q[2*V.NumModes:3*V.NumModes]) +\
              0.5*self.A.rho_inf*self.A.u_inf*self.A.c/2*self.AICsQhh[1,:,:].dot(q[:V.NumModes])

        if self.A.GustOn:
            #print self.gust_interpol(t)
            eta += self.gust_interpol(t)
            for i in range(self.A.NumPoles):
                eta += self.A.q_inf*q[(i+3)*V.NumModes:(i+4)*V.NumModes]
        else:
            for i in range(self.A.NumPoles):
                eta += self.A.q_inf*self.AICsQhh[i+3,:,:].dot(q[(i+3)*V.NumModes:(i+4)*V.NumModes])
        return eta

    def forceAero_eta_rbd(self,q,t):
        """Aerodynamic modal forces from modal exitacions, q1_dot removed, combining fixed rigid body components in clamped model"""
        #pdb.set_trace()
        if self.A.rbd_modify:
            if t>self.A.rbd_modify_time:
                q[2*V.NumModes+self.A.rbd:2*V.NumModes+2*self.A.rbd] = np.zeros(self.A.rbd)
        eta = self.A.q_inf*self.AICsQhh[0,:,:].dot(q[2*V.NumModes+self.A.rbd:3*V.NumModes+2*self.A.rbd]) +\
              0.5*self.A.rho_inf*self.A.u_inf*self.A.c/2*self.AICsQhh[1,:,:].dot(q[:V.NumModes+self.A.rbd])
        for i in range(self.A.NumPoles):
            eta += self.A.q_inf*self.AICsQhh[i+3,:,:].dot(q[(i+3)*V.NumModes+2*self.A.rbd+(i)*self.A.rbd:(i+4)*V.NumModes+2*self.A.rbd+(i+1)*self.A.rbd])
        return eta

    def forceAero_eta_x(self):
        #pdb.set_trace()
        eta = self.A.q_inf*self.AICsQx[:,0]*self.A.qx[0]
        for i in range(1,np.shape(self.AICsQx)[1]):
            eta += self.A.q_inf*self.AICsQx[:,i]*self.A.qx[i]
        return eta

    def forceAero_elevator(self,q_elevator):
        eta = np.zeros(V.NumModes)
        for i in range(len(self.A.aelink)):
            eta += self.A.q_inf*self.AICsQx[:,self.A.elevator_index[i]]*q_elevator*self.A.aelink[i]
        return eta

    def forceAero_spring(self,force_spring,direction_spring):
        eta=np.zeros(V.NumModes)
        f1 = self.BeamSeg[0].GlobalAxes.T.dot(force_spring*direction_spring)
        for k in range(V.NumModes):
          eta[k] = self.Phi1[0][k,0].dot(np.array([f1[0],f1[1],f1[2],0.,0.,0.]))
        return eta

    def forceAero_trim(self,t,q,rotation,args):
        #pdb.set_trace()
        control_times=0
        for tci in range(len(self.A.trim_time)):
            if (t >= self.A.trim_time[tci][0]) and (t <= self.A.trim_time[tci][1]):
                control_times+=1
                if tci>0:
                    q_elev0_on = 1
                else:
                    q_elev0_on = 0
                if abs(self.A.trim_time[tci][1]-t)<=V.dt:
                    self.q_elev0 = self.q_elevator[-1]
                break

        if control_times:
            displacement_tip = self.Phi1[-4][:,-1,:3].T.dot(q[2*V.NumModes:3*V.NumModes])
            displacement_local = self.Phi1[0][:,0,:3].T.dot(q[2*V.NumModes:3*V.NumModes])
            rotation_local = self.Phi1[0][:,0,3:].T.dot(q[2*V.NumModes:3*V.NumModes])
            displacement_global = self.BeamSeg[0].GlobalAxes.dot(displacement_local)
            rotation_global = self.BeamSeg[0].GlobalAxes.dot(rotation_local)
            if self.A.print_trim_cg:
                print 'rotation cg %s %s %s \n \n' %( rotation_global[0], rotation_global[1], rotation_global[2])
                print 'Displacement cg %s %s %s \n \n' %( displacement_global[0], displacement_global[1], displacement_global[2])
                #print 'Displacement tip %s %s %s \n \n' %( displacement_tip[0], displacement_tip[1], displacement_tip[2])

            velocity_local = self.Phi1[0][:,0,:3].T.dot(q[0:V.NumModes])
            velocity_global = self.BeamSeg[0].GlobalAxes.dot(velocity_local)
            omega_local = self.Phi1[0][:,0,3:].T.dot(q[0:V.NumModes])
            omega_global = self.BeamSeg[0].GlobalAxes.dot(omega_local)
            dt = (t-self.tm1)
            if self.A.Trim_NLin:
                Ipsi=((omega_local+self.omega_local)/2)*dt
                Itheta=np.linalg.norm(Ipsi)
                Rabi = self.Rabi.dot(H0(Itheta,Ipsi))
                rai = self.rai+self.Rabi.dot(H1(Itheta,Ipsi,dt)).dot((velocity_local+self.velocity_local)/2)
                if np.linalg.norm(rai) > 1e-5 and 0:
                    direction_spring=np.abs(rai)/np.linalg.norm(rai)
                else:
                    direction_spring= np.array([0.,0.,1])
                #pdb.set_trace()
                #force_spring = -self.A.K_spring*(np.linalg.norm([rai[0],rai[2]]))*np.sign(rai[2])
                force_spring = -self.A.K_spring*(rai[2])
                force_damper = -self.A.D_spring*((velocity_global)[2])**2*np.sign((velocity_global)[2])
                self.velocity_local = velocity_local
                self.omega_local = omega_local
                self.Rabi = Rabi
                self.rai = rai
                #print 'Displacement cg2 %s %s %s \n \n' %( rai[0], rai[1], rai[2])
            else:
                #force_spring = -self.A.K_spring*(rai[2])
                force_damper = -self.A.D_spring*((velocity_global)[2])**2*np.sign((velocity_global)[2])
                force_spring = -self.A.K_spring*(displacement_global)[2]
                direction_spring= np.array([0.,0.,1])

            #force_damper = -self.A.D_spring*((velocity_global)[2])**2*np.sign((velocity_global)[2])
            #print 'force_spring: %s \n \n' %force_spring
            if q_elev0_on:
                pdb.set_trace()
                q_elevator = self.q_elev0 + self.A.Gain_PID[0]*(force_spring+force_damper)+self.A.Gain_PID[2]*(self.force_spring_dot[-1]+self.force_damper_dot[-1])
            else:
                q_elevator = self.A.Gain_PID[0]*(force_spring+force_damper)+self.A.Gain_PID[1]*(self.force_spring_int[-1]+self.force_damper_int[-1])+self.A.Gain_PID[2]*(self.force_spring_dot[-1]+self.force_damper_dot[-1])
            if self.A.print_trim_PDIgains:
                print 'q_elevator: %s \n \n' %q_elevator
                #pdb.set_trace()
                print 'gain0: %s \n \n' % (self.A.Gain_PID[0]*(force_spring))
                print 'gain1: %s \n \n' % (self.A.Gain_PID[1]*self.force_spring_int[-1])
                print 'gain2: %s \n \n' % (self.A.Gain_PID[2]*self.force_spring_dot[-1])
            dt = (t-self.tm1)
            #pdb.set_trace()
            if dt > 0.:
                self.force_spring_dot.append((force_spring-self.force_spring[-1])/dt)
                self.force_damper_dot.append((force_damper-self.force_damper[-1])/dt)
            self.force_spring_int.append(self.force_spring_int[-1]+(force_spring)*dt)
            self.force_damper_int.append(self.force_damper_int[-1]+(force_damper)*dt)
            self.force_spring.append(force_spring)
            self.force_damper.append(force_damper)
            self.q_elevator.append(q_elevator)
            self.tm1 = t
            #pdb.set_trace()
            eta_elevator = self.forceAero_elevator(q_elevator)
            if (t>=self.A.time_attachment[0]) and (t<=self.A.time_attachment[1]):
                eta_spring = self.forceAero_spring(force_spring+force_damper,direction_spring)
            else:
                eta_spring = 0.
            if self.A.print_trim_eta:
                print 'eta_elevator: %s \n\n' %eta_elevator
                print 'eta_spring: %s \n' %eta_spring
        else:
             eta_elevator = self.forceAero_elevator(self.q_elevator[-1])
             eta_spring = 0.
             if self.A.print_trim_cg:
                 displacement_tip = self.Phi1[-4][:,-1,:3].T.dot(q[2*V.NumModes:3*V.NumModes])
                 displacement_local = self.Phi1[0][:,0,:3].T.dot(q[2*V.NumModes:3*V.NumModes])
                 rotation_local = self.Phi1[0][:,0,3:].T.dot(q[2*V.NumModes:3*V.NumModes])
                 displacement_global = self.BeamSeg[0].GlobalAxes.dot(displacement_local)
                 rotation_global = self.BeamSeg[0].GlobalAxes.dot(rotation_local)
                 print 'rotation cg %s %s %s \n \n' %( rotation_global[0], rotation_global[1], rotation_global[2])
                 print 'Displacement cg %s %s %s \n \n' %( displacement_global[0], displacement_global[1], displacement_global[2])
                 #print 'Displacement tip %s %s %s \n \n' %( displacement_tip[0], displacement_tip[1], displacement_tip[2])

        return eta_elevator+eta_spring

    def forceAeroStates(self,q1,t):

        #print self.gust_lags_interpol(t)
        lambdas = np.zeros((self.A.NumPoles,V.NumModes))
        for i in range(self.A.NumPoles):
            lambdas[i] = self.AICsQhh[i+3,:,:].dot(q1)+self.gust_lags_interpol(t)[i]

        return lambdas

    # def forceAero_gust(self,q,t):

    #     # qinf = 0.5*self.A.rho_inf*self.A.u_inf**2
    #     # eta = qinf*self.AICs[0,:,:].dot(q[2*V.NumModes:3*V.NumModes]) +\
    #     #       self.A.c*qinf/(2*self.A.u_inf)*self.AICs[1,:,:].dot(q[:V.NumModes]) +\
    #     #       qinf*(self.A.c/(2*self.A.u_inf))**2*self.AICs[2,:,:].dot(q[1*V.NumModes:2*V.NumModes])
    #     # for i in range(self.A.NumPoles):
    #     #     eta += qinf*self.AICs[i+3,:,:].dot(q[(i+3)*V.NumModes:(i+4)*V.NumModes])

    #     eta = 0.5*self.A.rho_inf*(self.A.u_inf**2*self.AICs[0,:,:].dot(q[2*V.NumModes:3*V.NumModes]) +
    #                               self.A.u_inf*self.A.c/2*self.AICs[1,:,:].dot(q[:V.NumModes]) +
    #                               (self.A.c/2)**2*self.AICs[2,:,:].dot(q[1*V.NumModes:2*V.NumModes]))
    #     for i in range(self.A.NumPoles):
    #         eta += 0.5*self.A.rho_inf*self.A.u_inf**2*self.AICs[i+3,:,:].dot(q[(i+3)*V.NumModes:(i+4)*V.NumModes])

    #     return eta


    def eta(self,t,q,rotation=None,args={}):

        eta = np.zeros(V.NumModes)
        if self.NumFLoads>0:
            eta += self.forceFollower_eta(t)

        if self.Gravity:
            #pdb.set_trace()
            eta += self.forceGravity_eta(t,rotation)
            print 'eta_gravity: %s \n \n' %eta
            if self.NumDLoads>0:
               #rotation =[]
               eta += self.forceDead_eta(t,rotation)
        elif self.NumDLoads>0:

            if rotation is not None:
                rotation_dead = rotation

            else:
                rotation_dead = [[q[2*(V.NumModes-V.NumModes_res)+l*4],q[2*(V.NumModes-V.NumModes_res)+l*4+1],q[2*(V.NumModes-V.NumModes_res)+l*4+2],q[2*(V.NumModes-V.NumModes_res)+l*4+3]] for l in range(self.NumDLoads)]

            eta += self.forceDead_eta(t,rotation_dead)

        if self.NumALoads>0:
            #pdb.set_trace()
            if self.A.rbd:
                eta = np.hstack([np.zeros(self.A.rbd),eta])
                eta += self.forceAero_eta_rbd(q,t)
            elif self.A.rbdx:
                eta += self.forceAero_etax(q)
                #if self.A.rbdx:
                eta += self.forceAero_eta_x()
            else:
                eta += self.forceAero_etax(q,t)
                print 'aero loads  %s \n \n' % self.forceAero_etax(q,t)
            if self.A.TrimOn:
                #eta += self.forceAero_etax(q)
                # if 'force_spring' not in args.keys():
                #     args['q_elevator'] = [0.]
                #     args['force_spring'] = [0.]
                # if 'force_spring_dot' not in args.keys():
                #     args['force_spring_dot'] = [0.]
                # if 'force_spring_int' not in args.keys():
                #     args['force_spring_int'] = [0.]
                # if 'tm1' not in args.keys():
                #     args['tm1'] = 0.
                eta += self.forceAero_trim(t,q,rotation,args)
        #print 'eta_total: %s \n\n' %eta
        return eta
#self.Ma[BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6,BeamSeg[i].NodeOrder[j]*6:BeamSeg[i].NodeOrder[j]*6+6].dot(Rab[i][j].T.dot(np.array([0.,0.,V.g,0.,0.,0.])))

# def Rotation4mStrain(Rab0,kappa,strain,V=V,BeamSeg=BeamSeg,inverseconn=inverseconn):
#     I3=np.eye(3)
#     e_1=np.array([1,0,0])
#     def IRab(Rab,Itheta,Ipsi):
#       return Rab.dot(H0(Itheta,Ipsi))

#     Rab=[[] for i in range(V.NumBeams)]

#     for i in range(V.NumBeams):

#         ra[i]=np.zeros((BeamSeg[i].EnumNodes,3))
#         Rab[i]=np.zeros((BeamSeg[i].EnumNodes,3,3))
#         if V.Clamped and i in V.BeamsClamped:
#             Rab[i][0]=BeamSeg[i].GlobalAxes
#         elif not V.Clamped and i in V.BeamsInit:
#             Rab[i][0]=Rab0
#         else:
#             k=inverseconn[i]
#             Rab[i][0] = BeamSeg[i].GlobalAxes.dot(BeamSeg[k].GlobalAxes.T.dot(Rab[k][-1]))
#             #Rab[i][0] = Rab[k][-1].dot(BeamSeg[k].GlobalAxes.T.dot(BeamSeg[i].GlobalAxes))

#         for j in range(1,BeamSeg[i].EnumNodes):
#           Ipsi=(kappa[i][j-1])*BeamSeg[i].NodeDL[j-1]
#           Itheta=np.linalg.norm(Ipsi)
#           Rab[i][j] = IRab(Rab[i][j-1],Itheta,Ipsi)

#     return(ra,Rab)


if __name__ == "__main__":
    # Phi1,Gravity=None,NumFLoads=0,NumDLoads=0,
    # Follower_points_app=None,Follower_interpol=None,
    # Dead_points_app=None,Dead_interpol=None
    NL=2
    X=[[0,1,[1,3]],[0,3,[0]]]
    I=[[[[0,5],[1,1]],[[0,5],[0,1]]],[[[0,5],[5,0]]]]
    t1 = Force(Phi1l,Follower_points_app=X,Follower_interpol=I,NumFLoads=NL)
    t2 = Force(Phi1l,Dead_points_app=X,Dead_interpol=I,NumDLoads=NL)

    NL=1
    X=[[0,-1,[1]]]
    I=[[[[0,1,2],[0,1,0]]]]
    t3 = Force(Phi1l,Follower_points_app=X,Follower_interpol=I,NumFLoads=NL)

    NL=1
    X=[[0,-1,[0,5]]]
    I=[[[[0.,2.5,2.5,10.],[8.,8.,0.,0.]],[[0.,2.5,2.5,10.],[80.,80.,0.,0.]]]]
    t3 = Force(Phi1l,Follower_points_app=X,Follower_interpol=I,NumFLoads=NL)

    NL = 1
    X = [[0,-1,[0,4,5]]]
    I = [[[[0.,2.5,5.,10.],[0.,20.,0.,0.]],[[0.,2.5,5.,10.],[0.,100.,0.,0.]],[[0.,2.5,5.,10.],[0.,200.,0.,0.]]]]
    t4 = Force(Phi1l,Follower_points_app=X,Follower_interpol=I,NumFLoads=NL)


class try1:
    def __init__(self,v1='0'):
       self.v1 = v1
    def fun(self):

        if self.v1=='0':
            def do1(self):
                print '0'
        elif self.v1 =='1':
            def do1(self):
                print '1'
        else:
            def do1(self):
                print '2'

    def print_v1(self):
        self.do1()
