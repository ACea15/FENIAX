from fem4inas.simulations.simulation import Simulation
import diffrax
class SerialSimulation(Simulation, cls_name="serial"):

    def init_systems(self):
        ...
        
    def trigger(self):
        # Implement trigger for SerialSimulation
        self._run_systems()

    def _run_systems(self):
        # Implement _run for SerialSimulation

        for k, sys in self.systems.items(): # only one item in the loop
            sys.set_ic()
            solver_sol = sys.solve()
            if self.settings.save_objs:
                self.sol.add_dict('dsys_sol', k, solver_sol)
            sys.pull_solution(self.sol)
            #self._post_run(qs, sol)

    def _post_run(self, q):
        # Implement _post_run for SerialSimulation
        pass

    def pull_solution(self):
        # Implement pull_solution for SerialSimulation
        pass
