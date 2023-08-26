from  fem4inas.systems.system import System
import fem4inas.systems.sollibs as sollibs
import fem4inas.intrinsic.dq as dq
import fem4inas.intrinsic.postprocess as postprocess
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsic
import fem4inas.preprocessor.solution as solution

class IntrinsicSystem(System, cls_name="intrinsic"):

    def __init__(self,
                 name: str,
                 settings: intrinsic.D_system,
                 fem: intrinsic.Dfem,
                 sol: solution.IntrinsicSolution):

        self.name = name
        self.settings = settings
        self.fem = fem
        self.sol = sol
        
    def set_ic(self):
        self.q0 = jnp.zeros(self.fem.num_modes)

    def set_name(self):
        pass

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

    def solve(self):

        args = (self.sol, )
        for ti in self.settings.t:
            
            sol = self.eqsolver(self.dFq,
                                self.q0,
                                args,
                                **self.settings.solver_settings)
            qi = self.states_puller(sol)
            

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
