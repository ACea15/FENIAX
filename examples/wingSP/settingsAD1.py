from  feniax.systems.system import System
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.dq_static as dq_static
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.intrinsic.postprocess as postprocess
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
import feniax.intrinsic.initcond as initcond
import feniax.intrinsic.args as libargs
import feniax.intrinsic.modes as modes
import feniax.intrinsic.couplings as couplings
import feniax.intrinsic.dq_common as common
import feniax.intrinsic.xloads as xloads
import feniax.intrinsic.objectives as objectives
import optimistix as optx
from functools import partial
import jax.numpy as jnp
import jax
import feniax.systems.sollibs.diffrax as libdiffrax
import feniax.systems.intrinsic_system as isys
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import pathlib

jax.config.update("jax_enable_x64", True)

import equinox


def _compute_modes(X,
                  Ka,
                  Ma,
                  eigenvals,
                  eigenvecs,
                  config):

    modal_analysis = modes.shapes(X.T,
                                  Ka,
                                  Ma,
                                  eigenvals,
                                  eigenvecs,
                                  config
                                  )

    return modes.scale(*modal_analysis)


@partial(jax.jit, static_argnames=['config', 'f_obj'])
def main_20g11(alpha,
               #t_array,
               q0,
               Ka,
               Ma,
               config,
               f_obj,
               obj_args=None
               ):

    if obj_args is None:
        obj_args = dict()

    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    q2_index = config.system.states['q2']
    q1_index = config.system.states['q1']
    eigenvals = jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = jnp.load(config.fem.folder / config.fem.eig_names[1])
    reduced_eigenvals = eigenvals[:config.fem.num_modes]
    reduced_eigenvecs = eigenvecs[:, :config.fem.num_modes]
    # solver_args = config.system.solver_settings
    X = config.fem.X
    (phi1, psi1, phi2,
     phi1l, phi1ml, psi1l, phi2l, psi2l,
     omega, X_xdelta, C0ab, C06ab) = _compute_modes(X,
                                                    Ka,
                                                    Ma,
                                                    reduced_eigenvals,
                                                    reduced_eigenvecs,
                                                    config)

    gamma1 = couplings.f_gamma1(phi1, psi1)
    gamma2 = couplings.f_gamma2(
        phi1ml,
        phi2l,
        psi2l,
        X_xdelta
    )
    config.system.xloads.build_point_follower(
                config.fem.num_nodes, C06ab)
    x_forceinterpol = config.system.xloads.x
    y_forceinterpol = alpha * config.system.xloads.force_follower
    states = config.system.states
    eta0 = jnp.zeros(config.fem.num_modes)
    dq_args = (eta0, gamma1, gamma2, omega, phi1,
               x_forceinterpol,
               y_forceinterpol, states)

    states_puller, eqsolver = sollibs.factory(config.system.solver_library,
                                              config.system.solver_function)

    sol = eqsolver(dq_dynamic.dq_20g11,
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
    # X2, X3, ra, Cab = recover_staticfields(q, tn, X, q2_index,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config.fem)
    tn = len(q)
    X1, X2, X3, ra, Cab = isys.recover_fields(q1,q2,
                                              tn, X,
                                              phi1l, phi2l,
                                              psi2l, X_xdelta,
                                              C0ab, config)
    # X2, X3, ra, Cab = isys.recover_staticfields(q2, tn, X,
    #                                        phi2l, psi2l, X_xdelta, C0ab, config)
    # X1 = jnp.zeros_like(X2)
    objective = f_obj(X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab, **obj_args)
    return objective

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path= pathlib.Path(
#     f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path= pathlib.Path(
    "./results_AD")

#inp.driver.sol_path=None
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "dynamic"
inp.system.t1 = 10.
inp.system.tn = 10001

inp.system.solver_library = "diffrax"
# inp.system.solver_library = "runge_kutta"
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="Dopri5")
# inp.system.solver_settings = dict(solver_name="rk4")
# inp.system.solver_library = "scipy"
# inp.system.solver_function = "root"
# inp.system.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
#inp.system.label = 'dq_101001'
inp.system.xloads.follower_forces = True
inp.system.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.system.xloads.x = [0, 4, 4+1e-6, 20]
inp.system.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                            [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                            ]
config =  configuration.Config(inp)

fprime = jax.value_and_grad(main_20g11)

F1, F1p  =fprime(1.,
               #t_array=jnp.array([1,2,3,4,5]), #jnp.array(config.system.t[:-1]),
               q0=jnp.zeros(config.fem.num_modes*2),
               Ka=config.fem.Ka,
               Ma=config.fem.Ma,
               config=config,
               # f_obj=objectives.OBJ_X2,
               # obj_args=dict(node=1,
               #              component=2),
               f_obj=objectives.OBJ_raMAX,
                 obj_args=dict(node=25,
                               component=1))

F2, F2p  =fprime(0.5,
               #t_array=jnp.array([1,2,3,4,5]), #jnp.array(config.system.t[:-1]),
               q0=jnp.zeros(config.fem.num_modes*2),
               Ka=config.fem.Ka,
               Ma=config.fem.Ma,
               config=config,
               f_obj=objectives.OBJ_X2,
               obj_args=dict(node=1,
                             component=2))

F3, F3p  =fprime(1.5,
               #t_array=jnp.array([1,2,3,4,5]), #jnp.array(config.system.t[:-1]),
               q0=jnp.zeros(config.fem.num_modes*2),
               Ka=config.fem.Ka,
               Ma=config.fem.Ma,
               config=config,
               f_obj=objectives.OBJ_X2,
               obj_args=dict(node=1,
                             component=2))


alpha1 = 1.
epsilon = 1e-4
F11 = main_20g11(alpha1,
           q0=jnp.zeros(config.fem.num_modes*2),
           Ka=config.fem.Ka,
           Ma=config.fem.Ma,
           config=config,
           f_obj=objectives.OBJ_X2,
           obj_args=dict(node=1,
                         component=2))
F12 = main_20g11(alpha1 + epsilon,
           q0=jnp.zeros(config.fem.num_modes*2),
           Ka=config.fem.Ka,
           Ma=config.fem.Ma,
           config=config,
           f_obj=objectives.OBJ_X2,
           obj_args=dict(node=1,
                         component=2))
F1dp = (F12 - F11) / epsilon
##############################################
alpha1 = 0.5
epsilon = 1e-4
F21 = main_20g11(alpha1,
           q0=jnp.zeros(config.fem.num_modes*2),
           Ka=config.fem.Ka,
           Ma=config.fem.Ma,
           config=config,
           f_obj=objectives.OBJ_X2,
           obj_args=dict(node=1,
                         component=2))
F22 = main_20g11(alpha1 + epsilon,
           q0=jnp.zeros(config.fem.num_modes*2),
           Ka=config.fem.Ka,
           Ma=config.fem.Ma,
           config=config,
           f_obj=objectives.OBJ_X2,
           obj_args=dict(node=1,
                         component=2))
F2dp = (F22 - F21) / epsilon
###############################################
alpha1 = 1.5
epsilon = 1e-4
F31 = main_20g11(alpha1,
           q0=jnp.zeros(config.fem.num_modes*2),
           Ka=config.fem.Ka,
           Ma=config.fem.Ma,
           config=config,
           f_obj=objectives.OBJ_X2,
           obj_args=dict(node=1,
                         component=2))
F32 = main_20g11(alpha1 + epsilon,
           q0=jnp.zeros(config.fem.num_modes*2),
           Ka=config.fem.Ka,
           Ma=config.fem.Ma,
           config=config,
           f_obj=objectives.OBJ_X2,
           obj_args=dict(node=1,
                         component=2))
F3dp = (F32 - F31) / epsilon



# (alpha=1) F1, F1p, F1dp => 3624434.84730555, 3735255.51062557, 3735117.42807925
# (alpha=0.5) F2, F2p, F2dp => 1723214.92320488, 3587711.70407052, 3587777.0972997
# (alpha=1.5) F3, F3p, F3dp => 5608253.89160295, 3957812.37570625, 3958312.94937059
