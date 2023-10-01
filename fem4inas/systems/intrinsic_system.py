from  fem4inas.systems.system import System
import fem4inas.systems.sollibs as sollibs
import fem4inas.intrinsic.dq as dq
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsic
import fem4inas.preprocessor.solution as solution
import fem4inas.intrinsic.initcond as initcond
import jax.numpy as jnp

class IntrinsicSystem(System, cls_name="intrinsic"):

    def __init__(self,
                 name: str,
                 settings: intrinsic.Dsystem,
                 fem: intrinsic.Dfem,
                 sol: solution.IntrinsicSolution,
                 config):

        self.name = name
        self.settings = settings
        self.fem = fem
        self.sol = sol
        self.config = config
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
                self.fem.num_nodes, self.sol.data.modes.C06ab)
            
    def set_states(self):
        self.settings.build_states(self.fem.num_modes)
        
    def set_generator(self):

        self.dFq = getattr(dq, self.settings.label)

    def set_solver(self):

        self.states_puller, self.eqsolver = sollibs.factory(
            self.settings.solver_library,
            self.settings.solver_function)

    def solve(self):

        self.state_sol = self.eqsolver(self.dFq,
                                       self.q0,
                                       **self.settings.solver_settings)

    def build_solution(self, sol: solution.IntrinsicSolution):

        qs = self.states_puller(self.state_sol)
        q1 = qs[self.settings.q1_index, :]
        q2 = qs[self.settings.q2_index, :]
        X1 = postprocess.compute_velocities(self.fem.phi1l, q1)
        X2 = postprocess.compute_internalforces(self.fem.phi2l, q2)
        X3 = postprocess.compute_strains(self.fem.cphi2l, q2)
        Rab = postprocess.velocity_Rab(X1)
        ra = postprocess.velocity_ra(X1, Rab)
        sol.add_container('DynamicSystem', label=self.name,
                          q=qs, X1=X1, X2=X2, X3=X3,
                          Rab=Rab, ra=ra)
    def save(self):
        pass

class StaticIntrinsic(IntrinsicSystem, cls_name="static_intrinsic"):

    # def _args_diffrax(self, t):

    #     return (t, self.sol, self.settings)

    # def _args_scipy(self, t):

    #     return ((t, self.sol, self.settings),)
    def _args_diffrax(self, t):

        return (t, self.sol, self.settings)

    def _args_scipy(self, t):

        return ((t, self.sol, self.settings),)

    def set_ic(self, q0):
        if q0 is None:
            self.q0 = jnp.zeros(self.fem.num_modes)
        else:
            self.q0 = q0

    def solve(self):

        solver_args = getattr(self, f"_args_{self.settings.solver_library}")
        qs = [self.q0]
        for i, ti in enumerate(self.settings.t):
            args1 = solver_args(ti)
            sol = self.eqsolver(self.dFq,
                                qs[-1],
                                args1,
                                **self.settings.solver_settings)
            qi = self.states_puller(sol)
            qs.append(qi)
        self.qs = jnp.array(qs[1:])

    def build_solution(self):

        # q1 = qs[self.settings.q1_index, :]
        # q2 = qs[self.settings.q2_index, :]
        tn = len(self.qs)
        ra0 = jnp.broadcast_to(self.fem.X[0], (tn, 3))
        Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
        X2 = postprocess.compute_internalforces(self.sol.data.modes.phi2l,
                                                self.qs)
        X3 = postprocess.compute_strains(self.sol.data.modes.psi2l,
                                         self.qs)
        Cab, ra = postprocess.integrate_strains_t(ra0,
                                                  Cab0,
                                                  X3,
                                                  self.sol,
                                                  self.fem
                                                  )
        self.sol.add_container('StaticSystem', label="_"+self.name,
                          q=self.qs, X2=X2, X3=X3,
                          Cab=Cab, ra=ra)
        if self.settings.save:
            self.sol.save_container('StaticSystem', label="_"+self.name)

    def build_solutionold(self):

        # q1 = qs[self.settings.q1_index, :]
        # q2 = qs[self.settings.q2_index, :]
        X2 = []
        X3 = []
        Cab = []
        ra = []
        for i, ti in enumerate(self.settings.t):
            X2t = postprocess.compute_internalforceso(self.sol.data.modes.phi2l, self.qs[i])
            X3t = postprocess.compute_strainso(self.sol.data.modes.psi2l, self.qs[i])
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

    def _args_diffrax(self):

        #return (self.sol, self.settings)
        return (self.sol, self.config)

    def _args_scipy(self):

        # return ((self.sol, self.settings),)
        return ((self.sol, self.config),)
    
    def _args_jax(self):

        #return (self.sol, self.settings)
        return (self.sol, self.config)

    def _args_runge_kutta(self):

        #return (self.sol, self.settings)
        return (self.sol.data.couplings.gamma1,
                self.sol.data.couplings.gamma2,
                self.sol.data.modes.omega,
                self.sol.data.modes.phi1l,
                self.settings.xloads.force_follower,
                self.settings.xloads.x,
                self.settings.states)

    def solve(self):

        solver_args = getattr(self, f"_args_{self.settings.solver_library}")
        args1 = solver_args()
        sol = self.eqsolver(self.dFq,
                            args1,
                            q0=self.q0,
                            t0=self.settings.t0,
                            t1=self.settings.t1,
                            tn=self.settings.tn,
                            dt=self.settings.dt,
                            t=self.settings.t,
                            **self.settings.solver_settings)
        self.qs = self.states_puller(sol)

    def build_solution(self):
        return
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
