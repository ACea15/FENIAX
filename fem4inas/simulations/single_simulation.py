from fem4inas.simulations.simulation import Simulation

class SingleSimulation(Simulation, cls_name="single"):
        
    def trigger(self):
        # Implement trigger for SerialSimulation
        self._run_systems()

    def _run_systems(self):
        # Implement _run for SerialSimulation

        for k, sys in self.systems.items(): # only one item in the loop
            sys.set_ic()
            sol = sys.solve()
            if self.settings.save_objs:
                self.sol.add_dict('dsys_sol', k, sol)
            sys.build_solution(self.sol)
            #self._post_run(qs, sol)

    def _save_states():
        ...
    def _post_run(self, qs, sol):
        # Implement _post_run for SerialSimulation
        ...
        

