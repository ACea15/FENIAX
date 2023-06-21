from abc import ABC, abstractmethod

class Simulation(ABC):
    @abstractmethod
    def trigger(self):
        pass

    @abstractmethod
    def _run(self):
        pass

    @abstractmethod
    def _post_run(self):
        pass

    @abstractmethod
    def pull_solution(self):
        pass

class SerialSimulation(Simulation):
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

class ParallelSimulation(Simulation):
    def trigger(self):
        # Implement trigger for ParallelSimulation
        pass

    def _run(self):
        # Implement _run for ParallelSimulation
        pass

    def _post_run(self):
        # Implement _post_run for ParallelSimulation
        pass

    def pull_solution(self):
        # Implement pull_solution for ParallelSimulation
        pass

class SingleSimulation(Simulation):
    def trigger(self):
        # Implement trigger for SingleSimulation
        pass

    def _run(self):
        # Implement _run for SingleSimulation
        pass

    def _post_run(self):
        # Implement _post_run for SingleSimulation
        pass

    def pull_solution(self):
        # Implement pull_solution for SingleSimulation
        pass

class CoupledSimulation(Simulation):
    def trigger(self):
        # Implement trigger for CoupledSimulation
        pass

    def _run(self):
        # Implement _run for CoupledSimulation
        pass

    def _post_run(self):
        # Implement _post_run for CoupledSimulation
        pass

    def pull_solution(self):
        # Implement pull_solution for CoupledSimulation
        pass
