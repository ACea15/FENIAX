from fem4inas.simulations.simulation import Simulation

class SingleSimulation(Simulation, cls_name="single"):
        
    def trigger(self):
        # Implement trigger for SerialSimulation
        self._run_systems()

    def _run_systems(self):
        # Implement _run for SerialSimulation

        for k, sys in self.systems.items():
            sys.set_ic()
            sol = sys.solve()
            qs = sys.pull_solution()
            

    def _post_run(self):
        # Implement _post_run for SerialSimulation
        pass

