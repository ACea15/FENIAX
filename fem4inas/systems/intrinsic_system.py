from  fem4inas.systems.system import System
import fem4inas.systems.sollibs as sollibs
import fem4inas.intrinsic.dq as dq

class IntrinsicSystem(System, cls_name="intrinsic"):
    
    def set_ic(self, q0):
        self.q0 = q0

    def set_name(self):
        pass

    def set_generator(self):

        self.dFq = getattr(dq, self.settings.label)

    def set_solver(self):

        self.eqsolver = sollibs.factory(
            self.settings.solver_library,
            self.settings.solver_function)

    def solve(self):

        sol = self.eqsolver(self.dFq,
                            self.q0,
                            **self.settings.solver_settings)
        return sol

        
    def save(self):
        pass
