from feniax.simulations.simulation import Simulation


class CoupledSimulation(Simulation, cls_name="coupled"):
    def trigger(self):
        # Implement trigger for SerialSimulation
        pass

    def _run(self):
        # Implement _run for SerialSimulation
        pass

    def _post_run(self):
        # Implement _post_run for SerialSimulation
        pass

    def pull_solution(self):
        # Implement pull_solution for SerialSimulation
        pass
