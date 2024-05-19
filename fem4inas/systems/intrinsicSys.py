from  fem4inas.systems.system import System
import fem4inas.systems.sollibs as sollibs
import fem4inas.intrinsic.dq_static as dq_static
import fem4inas.intrinsic.dq_dynamic as dq_dynamic
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsicmodal
import fem4inas.preprocessor.solution as solution
import fem4inas.intrinsic.initcond as initcond
import fem4inas.intrinsic.args as libargs
from functools import partial

import jax.numpy as jnp
import jax

def _staticSolve(eqsolver, dq, t_loads, q0, dq_args, sett):

    def _iter(qim1, t):
        args = dq_args + (t,)
        sol = eqsolver(dq, qim1, args, sett)
        qi = jnp.array(sol.value) #_static_optx(dq, qim1, args, config)
        return qi, qi

    qcarry, qs = jax.lax.scan(_iter,
                              q0,
                              t_loads)
    return qs

@partial(jax.jit, static_argnames=['config', 'tn'])
def recover_fields(q1, q2, tn, X,
                   phi1l, phi2l, psi2l, X_xdelta, C0ab, config):

    ra0 = jnp.broadcast_to(X[0], (tn, 3))
    Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
    X1 = postprocess.compute_velocities(phi1l,
                                        q1)
    
    X2 = postprocess.compute_internalforces(phi2l,
                                            q2)
    X3 = postprocess.compute_strains(psi2l,
                                     q2)
    Cab, ra = postprocess.integrate_strains_t(ra0,
                                              Cab0,
                                              X3,
                                              #fem,
                                              X_xdelta,
                                              C0ab,
                                              config
                                              )

    return X1, X2, X3, ra, Cab

@partial(jax.jit, static_argnames=['config', 'tn'])
def recover_staticfields(q2, tn, X, phi2l, psi2l, X_xdelta, C0ab, config):

    ra0 = jnp.broadcast_to(X[0], (tn, 3))
    Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
    X2 = postprocess.compute_internalforces(phi2l,
                                            q2)
    X3 = postprocess.compute_strains(psi2l,
                                     q2)
    Cab, ra = postprocess.integrate_strains_t(ra0,
                                              Cab0,
                                              X3,
                                              #fem,
                                              X_xdelta,
                                              C0ab,
                                              config
                                              )

    return X2, X3, ra, Cab


class IntrinsicSystem(System, cls_name="intrinsic"):

    def __init__(self,
                 name: str,
                 settings: intrinsicmodal.Dsystem,
                 config,
                 fem: intrinsicmodal.Dfem,
                 sol: solution.IntrinsicSolution):

        self.name = name
        self.settings = settings
        self.config = config
        self.fem = fem
        self.sol = sol
        #self._set_xloading()
        #self._set_generator()
        #self._set_solver()

    def set_ic(self, q0):
        if q0 is None:
            self.q0 = jnp.zeros(self.settings.num_states)
            if self.settings.init_states is not None:
                for k, v in self.settings.init_states.items():
                    if callable(v[0]):
                        init_f = v[0]
                    else:
                        init_f = getattr(initcond.Container, v[0])
                    x = init_f(*v[1], fem=self.fem) #6xNn inputs to approx.
                    # function to calculate qs
                    init_x = initcond.mapper[self.settings.init_mapper[k]]
                    sol_lstsq = init_x(self.sol.data.modes, x)
                    qi0 = sol_lstsq[0] # TODO: save to sol
                    self.q0 = self.q0.at[self.settings.states[k]].set(qi0)
        else:
            self.q0 = q0

    def set_xloading(self):
        if self.settings.xloads.follower_forces:
            self.settings.xloads.build_point_follower(
                self.fem.num_nodes, self.sol.data.modes.C06ab)
        if self.settings.xloads.dead_forces:
            self.settings.xloads.build_point_dead(
                self.fem.num_nodes)
        if self.settings.aero is not None:
            import fem4inas.intrinsic.aero as aero            
            approx = self.settings.aero.approx.capitalize()
            aeroobj = aero.Registry.create_instance(f"Aero{approx}",
                                                    self.settings,
                                                    self.sol)
            aeroobj.get_matrices()
            aeroobj.save_sol()
            if self.settings.aero.gust is not None:
                import fem4inas.intrinsic.gust as gust
                profile = self.settings.aero.gust_profile.capitalize()
                gustobj = gust.Registry.create_instance(f"Gust{approx}{profile}",
                                                        self.settings,
                                                        self.sol)
                gustobj.calculate_normals()
                gustobj.calculate_downwash()
                gustobj.set_solution(self.sol, self.settings.name)

    def set_states(self):
        self.settings.build_states(self.fem.num_modes)

    def set_solver(self):

        self.states_puller, self.eqsolver = sollibs.factory(
            self.settings.solver_library,
            self.settings.solver_function)

    def solve(self):

        self.state_sol = self.eqsolver(self.dFq,
                                       self.q0,
                                       **self.settings.solver_settings)

    def build_solution(self, sol: solution.IntrinsicSolution):

        ...

    def save(self):
        pass

class StaticIntrinsic(IntrinsicSystem, cls_name="static_intrinsic"):

    def set_ic(self, q0):
        if q0 is None:
            self.q0 = jnp.zeros(self.fem.num_modes)
        else:
            self.q0 = q0
            
    def set_system(self):

        label = self.settings.label
        print(f"***** Setting intrinsinc static system with label {label}")
        self.dFq = getattr(dq_static, label)

    def solve(self):

        label = self.settings.label.split("_")[-1]
        solver_args = getattr(libargs,
                              f"arg_{label}")
        args1 = solver_args(self.sol,
                            self.settings,
                            self.fem)

        self.qs =_staticSolve(self.eqsolver,
                              self.dFq,
                              self.settings.t,
                              self.q0,
                              args1,
                              self.settings.solver_settings)

    def solve_direct(self):

        label = self.settings.label.split("_")[-1]
        solver_args = getattr(libargs,
                              f"arg_{label}")
        args1 = solver_args(self.sol,
                            self.settings,
                            self.fem)

        def iterate(q0, ti):

            args = args1 + (ti,)
            sol = self.eqsolver(self.dFq,
                                q0,
                                args,
                                self.settings.solver_settings)
            qi = self.states_puller(sol)
            return qi, qi

        qcarry, self.qs = jax.lax.scan(iterate,
                                       self.q0,
                                       self.settings.t)

    def solve_forloop(self):

        label = self.settings.label.split("_")[-1]
        solver_args = getattr(libargs,
                              f"arg_{label}")
        qs = [self.q0]
        for i, ti in enumerate(self.settings.t):
            args1 = solver_args(self.sol,
                                self.settings,
                                self.fem)
            args = args1 + (ti,)
            sol = self.eqsolver(self.dFq,
                                qs[-1],
                                args,
                                **self.settings.solver_settings)
            qi = self.states_puller(sol)
            qs.append(qi)
        self.qs = jnp.array(qs[1:])

    def build_solution(self):

        # q1 = qs[self.settings.q1_index, :]
        q2_index = self.settings.states['q2']
        q2 = self.qs[:, q2_index]
        tn = len(self.qs)
        X2, X3, ra, Cab = recover_staticfields(q2,
                                               tn,
                                               self.fem.X,
                                               self.sol.data.modes.phi2l,
                                               self.sol.data.modes.psi2l,
                                               self.sol.data.modes.X_xdelta,
                                               self.sol.data.modes.C0ab,
                                               self.config)

        # ra0 = jnp.broadcast_to(self.fem.X[0], (tn, 3))
        # Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
        # X2 = postprocess.compute_internalforces(self.sol.data.modes.phi2l,
        #                                         self.qs)
        # X3 = postprocess.compute_strains(self.sol.data.modes.psi2l,
        #                                  self.qs)
        # Cab, ra = postprocess.integrate_strains_t(ra0,
        #                                           Cab0,
        #                                           X3,
        #                                           self.fem,
        #                                           self.sol.data.modes.X_xdelta,
        #                                           self.sol.data.modes.C0ab
        #                                           )
        self.sol.add_container('StaticSystem', label="_"+self.name,
                               q=self.qs, X2=X2, X3=X3,
                               Cab=Cab, ra=ra)
        if self.settings.save:
            self.sol.save_container('StaticSystem', label="_"+self.name)

    def build_solution_loop(self):

        # q1 = qs[self.settings.q1_index, :]
        # q2 = qs[self.settings.q2_index, :]
        X2 = []
        X3 = []
        Cab = []
        ra = []
        for i, ti in enumerate(self.settings.t):
            X2t = postprocess.compute_internalforces_t(self.sol.data.modes.phi2l, self.qs[i])
            X3t = postprocess.compute_strains_t(self.sol.data.modes.psi2l, self.qs[i])
            Cabt, rat = postprocess.integrate_strains(self.fem.X[0],
                                                      jnp.eye(3),
                                                      X3t,
                                                      self.sol,
                                                      self.fem
                                                      )
            X2.append(X2t)
            X3.append(X3t)
            Cab.append(Cabt)
            ra.append(rat)
            
        self.sol.add_container('StaticSystem', label="_"+self.name,
                          q=self.qs, X2=jnp.array(X2), X3=jnp.array(X3),
                          Cab=jnp.array(Cab), ra=jnp.array(ra))
        if self.settings.save:
            self.sol.save_container('StaticSystem', label="_"+self.name)

class DynamicIntrinsic(IntrinsicSystem, cls_name="dynamic_intrinsic"):

    def set_system(self):
        
        label = self.settings.label
        print(f"***** Setting intrinsinc Dynamic system with label {label}")
        self.dFq = getattr(dq_dynamic, label)

    def solve(self):

        label = self.settings.label.split("_")[-1]
        solver_args = getattr(libargs,
                              f"arg_{label}")
        args1 = solver_args(self.sol,
                            self.settings,
                            self.fem)
        sol = self.eqsolver(self.dFq,
                            args1,
                            self.settings.solver_settings,
                            q0=self.q0,
                            t0=self.settings.t0,
                            t1=self.settings.t1,
                            tn=self.settings.tn,
                            dt=self.settings.dt,
                            t=self.settings.t)
        self.qs = self.states_puller(sol)

    def build_solution_loop(self):
        #return
        # q1 = qs[self.settings.q1_index, :]
        # q2 = qs[self.settings.q2_index, :]
        X2 = []
        X3 = []
        Cab = []
        ra = []
        for i, ti in enumerate(self.settings.t):
            X2t = postprocess.compute_internalforces(self.sol.data.modes.phi2l, self.qs[i])
            X3t = postprocess.compute_strains(self.sol.data.modes.psi2l, self.qs[i])
            Cabt, rat = postprocess.integrate_strains(self.fem.X[0],
                                                      jnp.eye(3),
                                                      X3t,
                                                      self.sol,
                                                      self.fem
                                                      )
            X2.append(X2t)
            X3.append(X3t)
            Cab.append(Cabt)
            ra.append(rat)
            
        self.sol.add_container('DynamicSystem', label="_"+self.name,
                          q=self.qs, X2=jnp.array(X2), X3=jnp.array(X3),
                          Cab=jnp.array(Cab), ra=jnp.array(ra))
        if self.settings.save:
            self.sol.save_container('DynamicSystem', label="_"+self.name)

    def build_solution(self):

        q1_index = self.settings.states['q1']
        q2_index = self.settings.states['q2']
        q1 = self.qs[:, q1_index]
        q2 = self.qs[:, q2_index]
        tn = len(self.qs)
        X1, X2, X3, ra, Cab = recover_fields(q1,
                                             q2,
                                             tn,
                                             self.fem.X,
                                             self.sol.data.modes.phi1l,
                                             self.sol.data.modes.phi2l,
                                             self.sol.data.modes.psi2l,
                                             self.sol.data.modes.X_xdelta,
                                             self.sol.data.modes.C0ab,
                                             self.config
                                             )

        # ra0 = jnp.broadcast_to(self.fem.X[0], (tn, 3))
        # Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
        # X1 = postprocess.compute_velocities(self.sol.data.modes.phi1l,
        #                                     self.qs[:, self.settings.states['q1']])
        # X2 = postprocess.compute_internalforces(self.sol.data.modes.phi2l,
        #                                         self.qs[:, self.settings.states['q2']])
        # X3 = postprocess.compute_strains(self.sol.data.modes.psi2l,
        #                                  self.qs[:, self.settings.states['q2']])
        # Cab, ra = postprocess.integrate_strains_t(ra0,
        #                                           Cab0,
        #                                           X3,
        #                                           self.fem,
        #                                           self.sol.data.modes.X_xdelta,
        #                                           self.sol.data.modes.C0ab
        #                                           )
        
        self.sol.add_container('DynamicSystem', label="_" + self.name,
                               q=self.qs, X1=X1, X2=X2, X3=X3,
                               Cab=Cab, ra=ra, t=self.settings.t)
        if self.settings.save:
            self.sol.save_container('DynamicSystem', label="_"+self.name)
