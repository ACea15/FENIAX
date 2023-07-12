from fem4inas.simulations.simulation import Simulation

class SingleSimulation(Simulation, name="single"):

    def __init__(self, system):

        self.system = system
        
    def trigger(self):
        # Implement trigger for SerialSimulation
        pass

    def _run(self):
        # Implement _run for SerialSimulation

        self.system.set_ic()
        sol = self.system.solve()

    def _post_run(self):
        # Implement _post_run for SerialSimulation
        pass

    def pull_solution(self):
        # Implement pull_solution for SerialSimulation
        pass
