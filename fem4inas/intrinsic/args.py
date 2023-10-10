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
               t: float):

    gamma2 = sol.data.couplings.gamma2
    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower
    return (gamma2, omega, phi1, x,
            force_follower, t)

@catter2library
def arg_000001(sol: solution.IntrinsicSolution,
               system: intrinsic.Dsystem,
               t: float):

    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower
    return (omega, phi1, x,
            force_follower, t)

@catter2library
def arg_101000(sol: solution.IntrinsicSolution,
               system: intrinsic.Dsystem):

    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    states = system.states
    return gamma1, gamma2, omega, states

@catter2library
def arg_101001(sol: solution.IntrinsicSolution,
               system: intrinsic.Dsystem):

    phi1 = sol.data.modes.phi1l    
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower    
    states = system.states
    return (gamma1, gamma2, omega, phi1,
            x, force_follower, states)
