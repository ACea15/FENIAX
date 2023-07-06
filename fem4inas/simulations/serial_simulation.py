from fem4inas.simulations.simulation import Simulation

class SerialSimulation(Simulation, name="serial"):

    def init_systems(self):
        ...
        
    def trigger(self):
        # Implement trigger for SerialSimulation
        pass

    def _run(self):
        # Implement _run for SerialSimulation
        for si in self.systems:
            si.set_ic()
            q = si.solve()
            self._post_run(q)

    def _post_run(self, q):
        # Implement _post_run for SerialSimulation
        pass

    def pull_solution(self):
        # Implement pull_solution for SerialSimulation
        pass
