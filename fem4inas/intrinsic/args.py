import fem4inas.preprocessor.solution as solution
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsic

def _args_diffrax(input1):

    return input1

def _args_scipy(input1):

    return (input1,)

def _args_jax(self, input1):

    return input1

def _args_runge_kutta(input1):

    return input1

def catter2library(fun: callable):

    def wrapper(*args, **kwargs):

        args_ = fun(*args, **kwargs)
        solver_library = getattr(args[1],
                                 "solver_library")
        args_new = globals()[f"_args_{solver_library}"](args_)
        return args_new
    return wrapper 

@catter2library
def arg_001001(sol: solution.IntrinsicSolution,
               system: intrinsic.Dsystem,
               fem: intrinsic.Dfem,
               t: float,
               *args, **kwargs):

    gamma2 = sol.data.couplings.gamma2
    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower
    return (gamma2, omega, phi1, x,
            force_follower, t)

@catter2library
def arg_0011(sol: solution.IntrinsicSolution,
             system: intrinsic.Dsystem,
             fem: intrinsic.Dfem,
             t: float,
             *args, **kwargs):

    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    A0 = system.aero.A0
    B0 = system.aero.B0
    qx = system.aero.qx
    u_inf = system.aero.u_inf
    rho_inf = system.aero.rho_inf
    return (gamma2, omega,
            u_inf, rho_inf,
            qx, A0, B0)

@catter2library
def arg_000001(sol: solution.IntrinsicSolution,
               system: intrinsic.Dsystem,
               fem: intrinsic.Dfem,
               t: float,
               *args, **kwargs):

    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower
    return (omega, phi1, x,
            force_follower, t)

@catter2library
def arg_00101(sol: solution.IntrinsicSolution,
              system: intrinsic.Dsystem,
              fem: intrinsic.Dfem,
              t: float,
              *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l 
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_dead = system.xloads.force_dead
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma2, omega, phi1l, psi2l,
            x, force_dead,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father, t)

@catter2library
def arg_101000(sol: solution.IntrinsicSolution,
               system: intrinsic.Dsystem,
               *args, **kwargs):

    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    states = system.states
    return gamma1, gamma2, omega, states

@catter2library
def arg_101001(sol: solution.IntrinsicSolution,
               system: intrinsic.Dsystem,
               *args, **kwargs):

    phi1 = sol.data.modes.phi1l    
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower    
    states = system.states
    return (gamma1, gamma2, omega, phi1,
            x, force_follower, states)

@catter2library
def arg_100001(sol: solution.IntrinsicSolution,
               system: intrinsic.Dsystem,
               *args, **kwargs):

    phi1 = sol.data.modes.phi1l    
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower    
    states = system.states
    return (omega, phi1,
            x, force_follower, states)

@catter2library
def arg_10101(sol: solution.IntrinsicSolution,
              system: intrinsic.Dsystem,
              fem: intrinsic.Dfem,
              *args, **kwargs):

    phi1 = sol.data.modes.phi1l
    psi2 = sol.data.modes.psi2l 
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_dead = system.xloads.force_dead
    states = system.states
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma1, gamma2, omega, phi1, psi2,
            x, force_dead, states,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father)

