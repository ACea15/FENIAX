def scale():
    
    A0hat = q_inf * sol.data.modalaeroroger.A0
    D0hat = q_inf * sol.data.modalaeroroger.D0
    A1hat = (c_ref * rho_inf * u_inf / 4  *
             sol.data.modalaeroroger.A1)
    D1hat = (c_ref * rho_inf * u_inf / 4  *
             sol.data.modalaeroroger.D1)
    D2hat = (c_ref**2 * rho_inf / 8  *
             sol.data.modalaeroroger.D2)
    Aphat = q_inf * sol.data.modalaeroroger.Ap
    Dphat = q_inf * sol.data.modalaeroroger.Dp
