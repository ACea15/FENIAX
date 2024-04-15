from  fem4inas.systems.system import System
import scipy.linalg
import fem4inas.systems.sollibs as sollibs
import fem4inas.intrinsic.dq_static as dq_static
import fem4inas.intrinsic.dq_dynamic as dq_dynamic
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsicmodal
import fem4inas.preprocessor.solution as solution
import fem4inas.intrinsic.initcond as initcond
import fem4inas.intrinsic.args as libargs
import fem4inas.intrinsic.modes as modes
import fem4inas.intrinsic.couplings as couplings
import fem4inas.intrinsic.dq_common as common
import fem4inas.intrinsic.xloads as xloads
import fem4inas.intrinsic.objectives as objectives
import optimistix as optx
from functools import partial
import jax.numpy as jnp
import jax
import fem4inas.systems.sollibs.diffrax as diffrax
import fem4inas.systems.intrinsicSys as isys
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import pathlib

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import fem4inas.intrinsic.ad_common as adcommon
import fem4inas.intrinsic.gust as igust

@partial(jax.jit, static_argnames=['config', 'f_obj'])
def main_20g21(gust_intensity, gust_length, u_inf, rho_inf,
               q0,
               Ka,
               Ma,
               config,
               f_obj,
               obj_args=None
               ):

    # u_inf, rho_inf, gust_length, gust_intensity = input1
    if obj_args is None:
        obj_args = dict()

    config.system.build_states(config.fem.num_modes)
    q2_index = jnp.array(range(config.fem.num_modes, 2 * config.fem.num_modes)) #config.system.states['q2']
    q1_index = jnp.array(range(config.fem.num_modes)) #config.system.states['q1']
    #eigenvals, eigenvecs = scipy.linalg.eigh(Ka, Ma)
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0]).T
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1]).T
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    # solver_args = config.system.solver_settings
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = adcommon._compute_modes(X,
                                                             Ka,
                                                             Ma,
                                                             reduced_eigenvals,
                                                             reduced_eigenvecs,
                                                             config)
    #################
    gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(
        phi1ml,
        phi2l,
        psi2l,
        X_xdelta
    )

    #################
    A0 = config.system.aero.A[0]
    A1 = config.system.aero.A[1]
    A2 = config.system.aero.A[2]
    A3 = config.system.aero.A[3:]
    D0 = config.system.aero.D[0]
    D1 = config.system.aero.D[1]
    D2 = config.system.aero.D[2]
    D3 = config.system.aero.D[3:]    
    #u_inf = config.system.aero.u_inf
    #rho_inf = config.system.aero.rho_inf
    q_inf = 0.5 * rho_inf * u_inf ** 2 # config.system.aero.q_inf
    c_ref = config.system.aero.c_ref
    A0hat = q_inf * A0
    A1hat = c_ref * rho_inf * u_inf / 4 * A1
    A2hat = c_ref**2 * rho_inf / 8 * A2
    A3hat = q_inf * A3
    A2hatinv = jnp.linalg.inv(jnp.eye(len(A2hat)) -
                              A2hat)
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
    xgust = config.system.aero.gust.x
    time = config.system.aero.gust.time
    ntime = config.system.aero.gust.ntime
    # gust_totaltime, xgust, time, ntime, npanels = igust._get_gustRogerMc(
    #     gust_intensity,
    #     dihedral,
    #     gust_shift,
    #     gust_step,
    #     time,
    #     collocation_points,
    #     gust_length,
    #     u_inf)
    # gust_totaltime, xgust, time, ntime = igust._get_gustRogerMc(
    #     config.system.aero.gust.intensity,
    #     config.system.aero.gust.panels_dihedral,
    #     config.system.aero.gust.shift,
    #     config.system.aero.gust.step,
    #     time,
    #     config.system.aero.gust.length,
    #     config.system.aero.u_inf,
    #     jnp.min(collocation_points[:,0]),
    #     jnp.max(collocation_points[:,0])
    # )
    npanels = len(collocation_points)
    fshape_span = igust._get_spanshape(config.system.aero.gust.shape)
    gust, gust_dot, gust_ddot = igust._downwashRogerMc(u_inf,
                                                       gust_length,
                                                       gust_intensity,
                                                       gust_shift,
                                                       collocation_points,
                                                       dihedral, #normals,
                                                       time,
                                                       gust_totaltime,
                                                       fshape_span
                                                       )
    Q_w, Q_wdot, Q_wddot, Q_wsum, Ql_wdot = igust._getGAFs(D0hat,  # NbxNm
             D1hat,         
             D2hat,         
             D3hat,
             gust,
             gust_dot,
             gust_ddot
             )
    poles = config.system.aero.poles
    num_poles = config.system.aero.num_poles
    num_modes = config.fem.num_modes
    states = config.system.states
    dq_args = (gamma1, gamma2, omega, states,
               num_modes, num_poles,
               A0hat, A1hat, A2hatinv, A3hat,
               u_inf, c_ref, poles,
               time, Q_wsum, Ql_wdot)

    #################
    # import pdb;pdb.set_trace()
    states_puller, eqsolver = sollibs.factory(config.system.solver_library,
                                              config.system.solver_function)

    sol = eqsolver(dq_dynamic.dq_20g21,
                   dq_args,
                   config.system.solver_settings,
                   q0=q0,
                   t0=config.system.t0,
                   t1=config.system.t1,
                   tn=config.system.tn,
                   dt=config.system.dt,
                   t=config.system.t)
    q = states_puller(sol)
    
    #q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    # q = _solve(dq_static.dq_10g11, t_loads, q0, dq_args, config.system.solver_settings)
    q1 = q[:, q1_index]
    q2 = q[:, q2_index]
    tn = len(q)
    # X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    X1, X2, X3, ra, Cab = isys.recover_fields(q1,q2,
                                              tn, X,
                                              phi1l, phi2l,
                                              psi2l, X_xdelta,
                                              C0ab, config)
    # X1 = jnp.zeros_like(X2)
    objective = f_obj(X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab, **obj_args)
    return (ra[100, 2, 150]) #jnp.max(q1[:,0]) #objective

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_names = ["Dreal50.npy", "Vreal50.npy"]
#inp.fem.eig_type = "input_memory"
#inp.fem.eigenvals = jnp.load("./FEM/Dreal50.npy")
#inp.fem.eigenvecs = jnp.load("./FEM/Vreal50.npy").T
inp.fem.connectivity = [[1, 7, 13, 31], [2], [3], [4, 5], [27], [6], [],
                        [8], [9], [10, 11], [29], [12], [],
                        [14], [15], [16, 21], [17, 23, 25],
                        [18], [19], [20], [], [22], [], [24], [],
                        [26], [], [28], [], [30], [], []]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.grid = "structuralGridc.txt"
inp.fem.num_modes = 50
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path = pathlib.Path(
#     f"./resultsGust_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path = pathlib.Path(
    "./resultsGust_g1i2_m50")
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "dynamic"
inp.system.t1 = 0.1 #7.5
inp.system.tn = 10 #2001
inp.system.solver_library = "runge_kutta"
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="rk4")
inp.system.xloads.modalaero_forces = True
inp.system.q0treatment = 2
inp.system.aero.c_ref = 7.271
inp.system.aero.u_inf = 200.
inp.system.aero.rho_inf = 1.225
inp.system.aero.A = f"./NASTRAN/AERO/AICs081_8r{inp.fem.num_modes}.npy"
inp.system.aero.D = f"./NASTRAN/AERO/AICsQhj081_8r{inp.fem.num_modes}.npy"
inp.system.aero.poles = f"./NASTRAN/AERO/Poles081_8r{inp.fem.num_modes}.npy"
inp.system.aero.gust_profile = "mc"
inp.system.aero.gust.intensity = 14.0732311562*2 #11.304727674272842/10000
inp.system.aero.gust.length = 67.
inp.system.aero.gust.step = 1.
inp.system.aero.gust.shift = 0.
inp.system.aero.gust.panels_dihedral = jnp.load("./NASTRAN/AERO/Dihedral.npy")
inp.system.aero.gust.collocation_points = "./NASTRAN/AERO/Control_nodes.npy"

# path2config = pathlib.Path("./config.yaml")
config =  configuration.Config(inp)




fprime = jax.value_and_grad(main_20g21)

# Z=main_20g21(config.system.aero.u_inf,
#                  config.system.aero.rho_inf,
#                  config.system.aero.gust.length,
#                  config.system.aero.gust.intensity,
#                  #t_array=jnp.array([1,2,3,4,5]), #jnp.array(config.system.t[:-1]),
#                  q0=jnp.zeros(config.fem.num_modes*(2 + config.system.aero.num_poles)),
#                  Ka=config.fem.Ka,
#                  Ma=config.fem.Ma,
#                  config=config,
#                  f_obj=objectives.OBJ_ra,
#                  obj_args=dict(node=150,
#                                component=2)
#                  )


F1, F1p  =fprime(config.system.aero.gust.intensity,
                 config.system.aero.gust.length,
                 config.system.aero.u_inf,
                 config.system.aero.rho_inf,
                 #t_array=jnp.array([1,2,3,4,5]), #jnp.array(config.system.t[:-1]),
                 q0=jnp.zeros(config.fem.num_modes*(2 + config.system.aero.num_poles)),
                 Ka=config.fem.Ka,
                 Ma=config.fem.Ma,
                 config=config,
                 f_obj=objectives.OBJ_raMAX,
                 obj_args=dict(node=150,
                               component=2)
                 )

epsilon = 1e-4
F11 = main_20g21(config.system.aero.gust.intensity,
                 config.system.aero.gust.length,
                 config.system.aero.u_inf,
                 config.system.aero.rho_inf,
                 #t_array=jnp.array([1,2,3,4,5]), #jnp.array(config.system.t[:-1]),
                 q0=jnp.zeros(config.fem.num_modes*(2 + config.system.aero.num_poles)),
                 Ka=config.fem.Ka,
                 Ma=config.fem.Ma,
                 config=config,
                 f_obj=objectives.OBJ_raMAX,
                 obj_args=dict(node=150,
                               component=2)
                 )
F12 = main_20g21(config.system.aero.gust.intensity+epsilon,
                 config.system.aero.gust.length,
                 config.system.aero.u_inf,
                 config.system.aero.rho_inf,
                 #t_array=jnp.array([1,2,3,4,5]), #jnp.array(config.system.t[:-1]),
                 q0=jnp.zeros(config.fem.num_modes*(2 + config.system.aero.num_poles)),
                 Ka=config.fem.Ka,
                 Ma=config.fem.Ma,
                 config=config,
                 f_obj=objectives.OBJ_raMAX,
                 obj_args=dict(node=150,
                               component=2)
                 )
F1dp = (F12 - F11) / epsilon
