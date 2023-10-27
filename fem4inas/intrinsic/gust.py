class Gust:

    def __init__(self):
        ...
    def define_downwash(self):
        """
        NpxNt
        """
        ...
    def calculate_normals(self):
        ...
    def define_eta(self):
        """
        NtxNm
        """

class GustRogersMC:

    def __init__(self):
        ...
        
    def define_downwash(self):
        """
        NpxNt panel downwash in time
        """
        for panel in range (self.npanels):
            delay=(self.Control_nodes[panel,0]-self.A.X0_g)/self.A.u_inf
            shape_span = self.Gust_shape(self.Control_nodes[panel,1])
            if (time_>=self.A.Time_start_gust+delay and time_<=Time_finish_gust+delay):
                Gust[panel]=shape_span*self.Dihedral[panel]*(self.A.V0_g/(self.A.u_inf*2))*(1-np.cos(coeff2*(time_-self.A.Time_start_gust-delay)))
                Gust_dot[panel]=shape_span*self.Dihedral[panel]*(self.A.V0_g/(self.A.u_inf*2))*np.sin(coeff2*(time_-self.A.Time_start_gust-delay))*coeff2
                Gust_ddot[panel]=shape_span*self.Dihedral[panel]*(self.A.V0_g/(self.A.u_inf*2))*np.cos(coeff2*(time_-self.A.Time_start_gust-delay))*coeff2**2


                
    def define_eta(self):
        """
        NtxNm
        """
        ...

        Qgust = self.A.q_inf*np.dot(self.AICsQhj[0],Gust)
        Qgust_dot = self.A.q_inf*coeff1*np.dot(self.AICsQhj[1],Gust_dot)
        Qgust_ddot = self.A.q_inf*(coeff1**2)*np.dot(self.AICsQhj[2],Gust_ddot)
        Qgust_tot=Fgust+Fgust_dot+Fgust_ddot

