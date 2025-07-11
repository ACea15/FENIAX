import feniax.intrinsic.objectives as objectives
import optimistix as optx
from functools import partial
import jax.numpy as jnp
import jax
import feniax.intrinsic.modes as modes
import feniax.intrinsic.couplings as couplings
import feniax.systems.intrinsic_system as isys
import feniax.intrinsic.postprocess as postprocess
import equinox
import feniax.intrinsic.gust as igust


def _compute_modes(X, Ka, Ma, eigenvals, eigenvecs, config):
    modal_analysis = modes.shapes(X.T, Ka, Ma, eigenvals, eigenvecs, config)

    return modes.scale(*modal_analysis)

def _get_inputs(config, **kwargs):

    kwargs_list = list(kwargs.keys())
    if "Ka" in kwargs_list:
        Ka = kwargs.get("Ka")
    else:
        Ka = config.fem.Ka
    if "Ma" in kwargs_list:
        Ma = kwargs.get("Ma")
    else:
        Ma = config.fem.Ma
    if "eigenvals" in kwargs_list:
        eigenvals = kwargs.get("eigenvals")
    else:
        eigenvals = config.fem.eigenvals
    if "eigenvecs" in kwargs_list:
        eigenvecs = kwargs.get("eigenvecs")
    else:
        eigenvecs = config.fem.eigenvecs
    if "alpha" in kwargs_list:
        alpha = kwargs.get("alpha")
    else:
        alpha = 1.

    input_dict = dict(Ka=Ka, Ma=Ma, eigenvals=eigenvals,
                      eigenvecs=eigenvecs,
                      alpha=alpha
                      )
    return input_dict

def _get_inputs_aero(config, **kwargs):

    kwargs_list = list(kwargs.keys())
    if "Ka" in kwargs_list:
        Ka = kwargs.get("Ka")
    else:
        Ka = config.fem.Ka
    if "Ma" in kwargs_list:
        Ma = kwargs.get("Ma")
    else:
        Ma = config.fem.Ma
    if "eigenvals" in kwargs_list:
        eigenvals = kwargs.get("eigenvals")
    else:
        eigenvals = config.fem.eigenvals
    if "eigenvecs" in kwargs_list:
        eigenvecs = kwargs.get("eigenvecs")
    else:
        eigenvecs = config.fem.eigenvecs
    if "alpha" in kwargs_list:
        alpha = kwargs.get("alpha")
    else:
        alpha = 1.
    if "gust_intensity" in kwargs_list:
        gust_intensity = kwargs.get("gust_intensity")
    else:
        gust_intensity = config.system.aero.gust.intensity
    if "gust_length" in kwargs_list:
        gust_length = kwargs.get("gust_length")
    else:
        gust_length = config.system.aero.gust.length
    if "u_inf" in kwargs_list:
        u_inf = kwargs.get("u_inf")
    else:
        u_inf = config.system.aero.u_inf
    if "rho_inf" in kwargs_list:
        rho_inf = kwargs.get("rho_inf")
    else:
        rho_inf = config.system.aero.rho_inf

    input_dict = dict(Ka=Ka, Ma=Ma, eigenvals=eigenvals,
                      eigenvecs=eigenvecs,
                      alpha=alpha,
                      gust_intensity=gust_intensity,
                      gust_length=gust_length, u_inf=u_inf, rho_inf=rho_inf
                      )
    return input_dict

def _build_intrinsic(get_inputs, config, **kwargs):

    input_dict = get_inputs(config, **kwargs)
    X = config.fem.X
    (
        phi1,
        psi1,
        phi2,
        phi1l,
        phi1ml,
        psi1l,
        phi2l,
        psi2l,
        omega,
        X_xdelta,
        C0ab,
        C06ab
    ) = _compute_modes(X,
                       input_dict['Ka'],
                       input_dict['Ma'],
                       input_dict['eigenvals'],
                       input_dict['eigenvecs'],
                       config)
    gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(phi1ml, phi2l, psi2l, X_xdelta)
    output_dict = dict(phi1 = phi1,
                       psi1 = psi1,
                       phi2 = phi2,
                       phi1l = phi1l,
                       phi1ml = phi1ml,
                       psi1l = psi1l,
                       phi2l = phi2l,
                       psi2l = psi2l,
                       omega = omega,
                       X_xdelta = X_xdelta,
                       C0ab = C0ab,
                       C06ab = C06ab,
                       gamma1=gamma1,
                       gamma2 = gamma2,
                       )
    return output_dict, input_dict

def _build_solution(q, output_dict, config):

    X = config.fem.X
    tn = len(q)
    q1_index = config.system.states["q1"]                                      
    q2_index = config.system.states["q2"]
    q1 = q[:, q1_index]
    q2 = q[:, q2_index]
    X1, X2, X3, ra, Cab = isys.recover_fields(
        q1,
        q2,
        tn,
        X,
        output_dict['phi1l'],
        output_dict['phi2l'],
        output_dict['psi2l'],
        output_dict['X_xdelta'],
        output_dict['C0ab'],
        config
    )
    output_dict['q'] = q
    output_dict['X1'] = X1
    output_dict['X2'] = X2
    output_dict['X3'] = X3    
    output_dict['ra'] = ra
    output_dict['Cab'] = Cab

def _build_solutionRB(q, output_dict, config):

    X = config.fem.X
    tn = config.system.tn #len(q) WARNING: needs to be static for the recover 
    dt = config.system.dt 
    q1_index = config.system.states["q1"]                                      
    q2_index = config.system.states["q2"]
    q1 = q[:, q1_index]
    q2 = q[:, q2_index]
    X1, X2, X3, ra, Cab = isys.recover_fieldsRB(
        q1,
        q2,
        tn,
        dt,
        X,
        output_dict['phi1l'],
        output_dict['phi2l'],
        output_dict['psi2l'],
        output_dict['X_xdelta'],
        output_dict['C0ab'],
        config
    )
    output_dict['q'] = q
    output_dict['X1'] = X1
    output_dict['X2'] = X2
    output_dict['X3'] = X3    
    output_dict['ra'] = ra
    output_dict['Cab'] = Cab

    
def _get_aero(u_inf, rho_inf, config):

    q_inf = 0.5 * rho_inf * u_inf ** 2 
    A0 = config.system.aero.A[0]
    A1 = config.system.aero.A[1]
    A2 = config.system.aero.A[2]
    A3 = config.system.aero.A[3:]    
    c_ref = config.system.aero.c_ref
    poles = config.system.aero.poles
    A0hat = q_inf * A0
    A1hat = c_ref * rho_inf * u_inf / 4 * A1
    A2hat = c_ref**2 * rho_inf / 8 * A2
    A3hat = q_inf * A3
    A2hatinv = jnp.linalg.inv(jnp.eye(len(A2hat)) - A2hat)
    return (q_inf,
            c_ref,
            poles,
            A0hat,
            A1hat,
            A2hatinv,
            A3hat
            )
def _get_aerogust(config):

    A = config.system.aero.A
    D = config.system.aero.D
    c_ref = config.system.aero.c_ref
    poles = config.system.aero.poles
    num_poles = config.system.aero.num_poles
    xgust = config.system.aero.gust.time
    return (A,
            D,
            c_ref,
            poles,
            num_poles,
            xgust
            )
    
def _get_gust(input_dict, q_inf, c_ref, config):

    u_inf = input_dict['u_inf']
    rho_inf = input_dict['rho_inf']
    gust_intensity = input_dict['gust_intensity']
    gust_length = input_dict['gust_length']
    
    D0 = config.system.aero.D[0]
    D1 = config.system.aero.D[1]
    D2 = config.system.aero.D[2]
    D3 = config.system.aero.D[3:]    
    D0hat = q_inf * D0
    D1hat = c_ref * rho_inf * u_inf / 4 * D1
    D2hat = c_ref**2 * rho_inf / 8 * D2
    D3hat = q_inf * D3
    # gust_intensity = config.system.aero.gust.intensity
    # gust_length = config.system.aero.gust.length
    gust_shift = config.system.aero.gust.shift
    gust_step = config.system.aero.gust.step
    dihedral = config.system.aero.gust.panels_dihedral
    time = config.system.t
    collocation_points = config.system.aero.gust.collocation_points
    gust_totaltime = config.system.aero.gust.totaltime
    timegust = config.system.aero.gust.time
    ntime = config.system.aero.gust.ntime

    npanels = len(collocation_points)
    fshape_span = igust._get_spanshape(config.system.aero.gust.shape)
    gust, gust_dot, gust_ddot = igust._downwashRogerMc(
        u_inf,
        gust_length,
        gust_intensity,
        gust_shift,
        collocation_points,
        dihedral,  # normals,
        timegust,
        gust_totaltime,
        fshape_span,
    )
    Q_w, Q_wdot, Q_wddot, Q_wsum, Ql_wdot = igust._getGAFs(
        D0hat,  # NbxNm
        D1hat,
        D2hat,
        D3hat,
        gust,
        gust_dot,
        gust_ddot,
    )
    return timegust, Q_wsum, Ql_wdot


#@partial(jax.jit, static_argnames=["f_obj"])
def _objective_output(sol_dict, f_obj, *args, **kwargs):
    obj = f_obj(**sol_dict, **kwargs)
    #objective = jnp.hstack(jnp.hstack(obj))
    # objective = f_obj(X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab, axis=axis, **kwargs)
    # lax cond or select not working here as both branches are evaluated.
    # made sure obj is an array in f_obj
    # objective = jax.lax.select(len(obj.shape) > 0, jnp.hstack(obj), obj)
    # objective = jax.lax.cond(len(obj.shape) > 0,
    #                          lambda x : jnp.hstack(x),
    #                          lambda x : x, obj)
    #f_out = (objective, q, X1, X2, X3, ra, Cab)
    f_out = sol_dict | dict(objective=obj)
    return (obj, f_out)
