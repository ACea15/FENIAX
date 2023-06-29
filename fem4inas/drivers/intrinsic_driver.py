import fem4inas.intrinsic.modes as modes
import fem4inas.simulations

class IntrinsicDriver:
    def __init__(self):
        self.case = None
        self.integration = None
        self.simulation = None

    def pre_simulation(self):
        self._compute_modalshapes()
        
    def _compute_modalshapes(self):
        modes.Smodes =  modes.shapes(args, kwargs)
        
    def _compute_modalcouplings(self):
        ...
    def _load_modalshapes(self):
        ...
    def _load_modalcouplings(self):
        ...

    def run_cases(self):
        self.set_case()
        self.set_integration()
        self.set_simulation()

        self.simulation.trigger()
        # Perform other operations using the configured case, integration, and simulation

    def set_case(self):
        # Configure the case
        pass

    def set_integration(self):
        # Configure the integration
        pass

    def set_simulation(self):
        # Configure the simulation
        cls_simulation = fem4inas.simulations.factory_method(config.simulation.typeof)
        self.simulation = cls_simulation()
