from  fem4inas.systems.system import System
import fem4inas.systems.sollibs as sollibs
import fem4inas.intrinsic.dq as dq
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsic
import fem4inas.preprocessor.solution as solution
import jax.numpy as jnp

class IntrinsicSystem(System, cls_name="intrinsic"):

    def __init__(self,
                 name: str,
                 settings: intrinsic.Dsystem,
                 fem: intrinsic.Dfem,
                 sol: solution.IntrinsicSolution):

        self.name = name
        self.settings = settings
        self.fem = fem
        self.sol = sol
        #self._set_xloading()
        #self._set_generator()
        #self._set_solver()

    def set_ic(self, q0):
        self.q0 = q0

    def set_xloading(self):
        if self.settings.xloads.follower_forces:
            self.settings.xloads.build_point_follower(
                self.fem.num_nodes, self.sol.data.modes.C06ab)
        if self.settings.xloads.dead_forces:
            self.settings.xloads.build_point_dead(
                self.fem.num_nodes, self.sol.data.modes.C06ab)

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

    def _args_diffrax(self, t):

        return (t, self.sol, self.settings)

    def _args_scipy(self, t):

        return ((t, self.sol, self.settings),)

    def solve(self):

        solver_args = getattr(self, f"_args_{self.settings.solver_library}")
        qs = [jnp.zeros(self.fem.num_modes)]
        for i, ti in enumerate(self.settings.t):
            args1 = solver_args(ti)
            sol = self.eqsolver(self.dFq,
                                qs[-1],
                                args1,
                                **self.settings.solver_settings)
            qi = self.states_puller(sol)
            qs.append(qi)
        self.qs = jnp.array(qs)

    def build_solution(self, sol: solution.IntrinsicSolution):

        # q1 = qs[self.settings.q1_index, :]
        # q2 = qs[self.settings.q2_index, :]
        X2 = []
        X3 = []
        Cab = []
        ra = []
        for i in range(len(self.settings.t) + 1):
            X2t = postprocess.compute_internalforces(self.sol.data.modes.phi2l, self.qs[i])
            X3t = postprocess.compute_strains(self.sol.data.modes.psi2l, self.qs[i])
            Cabt, rat = postprocess.integrate_strains(jnp.zeros(3),
                                                      jnp.eye(3),
                                                      X3t,
                                                      self.sol,
                                                      self.fem
                                                      )
            X2.append(X2t)
            X3.append(X3t)
            Cab.append(Cabt)
            ra.append(rat)
            
        sol.add_container('StaticSystem', label="_"+self.name,
                          q=self.qs, X2=jnp.array(X2), X3=jnp.array(X3),
                          Cab=jnp.array(Cab), ra=jnp.array(ra))
