from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

class Case(ABC):
    pass

class Integration(ABC):
    pass

class Simulation(ABC):
    pass

class Driver:
    def __init__(self):
        self.case = None
        self.integration = None
        self.simulation = None

    def run_cases(self):
        self.set_case()
        self.set_integration()
        self.set_simulation()
        # Perform other operations using the configured case, integration, and simulation

    def set_case(self):
        # Configure the case
        pass

    def set_integration(self):
        # Configure the integration
        pass

    def set_simulation(self):
        # Configure the simulation
        pass

class DIntrinsicModal:

    def build():
        ...
        # calculate_modes
        # calculate_tensors

    def run():
        ...

    def postprocess():
        ...
    
