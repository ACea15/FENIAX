import      os
import      numpy      as        np
import scipy


class Gust4Nastran:

    def __init__(self,h,M,U_inf_TAS,Type,chord,F_max,N_step_freq,H,Fg,grid_DAREA,VD=False):

        # INPUT:
        self.h = h
        self.F_max = F_max
        self.N_step_freq = N_step_freq
        self.M = M
        self.chord = chord
        self.VD=VD
        self.U_inf_TAS = U_inf_TAS
        self.Type = Type
        self.H = H
        self.Fg = Fg
        self.grid_DAREA = grid_DAREA
        #MLW=78300.
        #MTOW=90300.
        #MZFW=74800.
        #Zmo=12496.8      #41000ft

        # Constant definition
        T_0=288.16
        rho_0=1.225
        P_0=1013254
        k=0.0065
        g=9.806
        R=287.05
        gamma=1.4

        self.T,self.rho,self.P,self.a = self.standard_atmosphere(self.h,k,R,g,gamma,T_0,rho_0)
        self.TAS2EAS=scipy.sqrt(self.rho/rho_0)
        U_inf_EAS=self.TAS2EAS*self.U_inf_TAS

        # Frequency and Time step Definition
        self.delta_f=F_max/N_step_freq
        self.T_max=1/self.delta_f                   # Max. time (3 times the real one) 
        self.delta_T=1/(F_max*4)                    # Time increments, dT 
        self.N_step_T=int(self.T_max/self.delta_T)  #Number of time steps (3*real_time_step)

        # Alleviation Factor Definition
        #R_1=MLW/MTOW
        #R_2=MZFW/MTOW
        #Fgz=1.-Zmo/76200.
        #Fgm=sqrt(R_2*tan(pi*R_1/4))
        #Fg_0=0.5*(Fgz+Fgm)
        #Fg_41k=1.
        #Fg=(Fg_41k-Fg_0)/(Zmo)*h+Fg_0
        #Fg=1.
        # Gust Family Definition
        #H=zeros(10)
        #H[:]=[9.100,19.978,30.856,41.733,52.611,63.489,74.367,85.244,96.122,107.000] #Andrea PhD
        #H=zeros(9)
        #H[:]=[9.144,15.24,21.336,30.48,45.72,60.96,76.2,91.44,106.68] #Airbus 1mc family

        # U_ref Definition in EAS
        if 0.<=self.h<=4575.:
            self.U_ref=-0.00080052*self.h+17.07
        if 4575.<=self.h<=18288.:
            self.U_ref=-0.000514*self.h+15.76
        if self.VD==True:
            self.U_ref=self.U_ref/2.

        self.U_ds = []
        self.delta_T_gust = []
        for i in range(len(H)):
            self.U_ds.append(self.U_ref*self.Fg*(self.H[i]/106.68)**(1./6.))
            if self.Type=='tas':
                self.U_inf=self.U_inf_TAS
                self.U_ds[i]=self.U_ds[i]/self.TAS2EAS
            if self.Type=='eas':
                self.U_inf=self.U_inf_EAS  
                self.rho=rho_0
            self.delta_T_gust.append(2.*self.H[i]/self.U_inf)

        self.q_dyn=0.5*self.rho*self.U_inf**2            

    def standard_atmosphere(self,h,k,R,g,gamma,T_0,rho_0):
        n=1/(1-k*R/g)
        if h< 11000.:
            T=T_0-k*h
            rho=rho_0*(T/T_0)**(1/(n-1))
            P=rho*R*T
            a=scipy.sqrt(gamma*R*T)
        elif 11000.<=h<=25000.:
            h_11k=11000.
            T_11k=T_0-k*h_11k
            rho_11k=rho_0*(T_11k/T_0)**(1/(n-1))
            P_11k=rho_11k*R*T_11k

            psi=scipy.exp(-(h-h_11k)*g/(R*T_11k))
            T=T_11k
            rho=rho_11k*psi
            P=P_11k*psi
            a=scipy.sqrt(gamma*R*T_11k)
        return T,rho,P,a

    def gust_family(self,write_gust,write_subcase):

        Gust_Card_file=open(write_gust,'w')
        subcase_file=open(write_subcase,'w')

        for i in range (len(self.H)):
            subcase_file.write('SUBCASE'.ljust(9)+str(int(i+1)).ljust(8))
            subcase_file.write('\n')
            subcase_file.write(' GUST  = '+str(int((i+1)*10)).ljust(8))
            subcase_file.write('\n')
            subcase_file.write(' DLOAD = '+str(int(i+1)).ljust(8))
            subcase_file.write('\n')
            subcase_file.write(' DISP = 1')
            subcase_file.write('\n')
            Gust_Card_file.write('TLOAD1'.ljust(8)+str(int(100*(i+1))).ljust(8)+'99999999'+' '.ljust(16)+str(int(i+1)).ljust(8))
            Gust_Card_file.write('\n')
            Gust_Card_file.write('DLOAD'.ljust(8)+str(int(i+1)).ljust(8)+'1.'.ljust(8)+'1.'.ljust(8)+str(int(100*(i+1))).ljust(8))
            Gust_Card_file.write('\n')
            Gust_Card_file.write('GUST'.ljust(8)+str(int((i+1)*10)).ljust(8)+str(int(i+1)).ljust(8)+((str("%.7f" % float(1./self.U_inf))).lstrip('0')).ljust(8)+'0.'.ljust(8)+('%.7s' % str(float(self.U_inf))).ljust(8))
            Gust_Card_file.write('\n')
            Gust_Card_file.write('TABLED1'.ljust(8)+str(int(i+1)).ljust(8))
            Gust_Card_file.write('\n')
            counter_t=0
            while counter_t<= self.N_step_T:
                end_check=False
                for col in range (5): 
                    if col==0:
                        Gust_Card_file.write(''.ljust(8))
                    else:
                        if counter_t<= self.N_step_T:
                            time=self.delta_T*counter_t
                            Gust_Card_file.write(('%.7s' % str(time)).ljust(8))
                            delta_T_gust=2.*self.H[i]/self.U_inf # Time to get passed the gust
                            if time<=delta_T_gust:
                                U=self.U_ds[i]/2*(1-np.cos(np.pi*self.U_inf*time/self.H[i]))
                            if time>delta_T_gust and time<=self.T_max:
                                U=0.
                            if time>self.T_max/3 and time<=self.T_max/3+delta_T_gust :
                                U=-2*self.U_ds[i]/2*(1-np.cos(np.pi*self.U_inf*(time-self.T_max/3)/self.H[i]))

                            if time>self.T_max/3+delta_T_gust and time<=self.T_max*2/3:
                                U=0.
                            if time>self.T_max*2/3 and time<=self.T_max*2/3+delta_T_gust:
                                U=self.U_ds[i]/2*(1-np.cos(np.pi*self.U_inf*(time-self.T_max*2/3)/self.H[i]))
                            if time>self.T_max*2/3+delta_T_gust:
                                U=0.
                            Gust_Card_file.write(('%.7s' % str("%.7f" % U)).ljust(8))
                            counter_t=counter_t+1
                            if (counter_t==self.N_step_T+1) and (col<4):
                                Gust_Card_file.write('ENDT'.ljust(8))
                                end_check=True
                Gust_Card_file.write('\n')
            if end_check==False:
                Gust_Card_file.write(' '.ljust(8)+'ENDT'.ljust(8))
                Gust_Card_file.write('\n')
        Gust_Card_file.close()
        subcase_file.close()  


    def gust_setup(self,setup_file):
        Gust_setup_file=open(setup_file,'w')
        Gust_setup_file.write('PARAM   Q'.ljust(16)+('%.7s' % str(float(self.q_dyn))).ljust(8))
        Gust_setup_file.write('\n')
        Gust_setup_file.write('PARAM   MACH'.ljust(16)+('%.7s' % str(float(self.M))).ljust(8))
        Gust_setup_file.write('\n')
        Gust_setup_file.write('AERO            '.ljust(16)+('%.7s' % str(float(self.U_inf))).ljust(8)+str(self.chord).ljust(8)+('%.7s' % str(float(self.rho))).ljust(8)+'0       0 '.ljust(24))
        Gust_setup_file.write('\n')
        Gust_setup_file.write('FREQ1'.ljust(8)+'930'.ljust(8)+'0.'.ljust(8)+('%.7s' % str(float(self.delta_f))).ljust(8)+('%.7s' % str(self.N_step_freq)).ljust(8))
        Gust_setup_file.write('\n')
        Gust_setup_file.write('TSTEP'.ljust(8)+'940'.ljust(8)+('%.7s' % str(int((self.N_step_T/3)))).ljust(8)+('%.7s' % str(float(self.delta_T))).ljust(8) +('%.7s' % str(1)).ljust(8))
        Gust_setup_file.write('\n')
        Gust_setup_file.write('DAREA'.ljust(8)+'99999999'.ljust(8)+('%s'%self.grid_DAREA).ljust(8)+'3'.ljust(8)+'1.'.ljust(8))
        Gust_setup_file.write('\n')
        Gust_setup_file.write('MKAERO1'.ljust(8)+('%.7s' % str(self.M)).ljust(8))     
        Gust_setup_file.write('\n')
        Gust_setup_file.write('        0.00001 0.0001  0.001   0.01    0.015   0.02    0.03    0.04')              
        Gust_setup_file.write('\n')
        Gust_setup_file.write('MKAERO1'.ljust(8)+('%.7s' % str(self.M)).ljust(8))          
        Gust_setup_file.write('\n')
        Gust_setup_file.write('        0.05    0.06    0.07    0.08    0.1     0.13    0.16    0.19')      
        Gust_setup_file.write('\n')
        Gust_setup_file.write('MKAERO1'.ljust(8)+('%.7s' % str(self.M)).ljust(8))          
        Gust_setup_file.write('\n')
        Gust_setup_file.write('        0.23    0.28    0.33    0.38    0.43    0.48    0.53    0.59')      
        Gust_setup_file.write('\n')
        Gust_setup_file.write('MKAERO1'.ljust(8)+('%.7s' % str(self.M)).ljust(8))          
        Gust_setup_file.write('\n')
        Gust_setup_file.write('        0.65    0.71    0.77    0.84    0.91    0.98    1.05    1.12')      
        Gust_setup_file.write('\n')
        Gust_setup_file.write('MKAERO1'.ljust(8)+('%.7s' % str(self.M)).ljust(8))          
        Gust_setup_file.write('\n')
        Gust_setup_file.write('        1.2     1.3     1.4     1.5     1.6     1.75    1.85    2.00')      
        Gust_setup_file.write('\n')

        Gust_setup_file.close()            

        Gust_setup_file=open('.'+setup_file.split('.')[-2]+'_var.'+setup_file.split('.')[-1],'w')

        Gust_setup_file.write('L_g                          ')       
        for i in range (len(self.U_ds)):
            Gust_setup_file.write(str(self.H[i]*2))
            if i<len(self.U_ds)-1:
                Gust_setup_file.write(',')      
        Gust_setup_file.write('\n')
        Gust_setup_file.write('V0_g                         ')
        for i in range (len(self.U_ds)):
            Gust_setup_file.write(str(self.U_ds[i]))
            if i<len(self.U_ds)-1:
                Gust_setup_file.write(',')     
        Gust_setup_file.close()
