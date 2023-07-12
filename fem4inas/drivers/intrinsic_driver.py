import fem4inas.intrinsic.modes as modes
import fem4inas.intrinsic.couplings as couplings

import fem4inas.simulations
import fem4inas.integrations.integration
from fem4inas.preprocessor import solution, config

class IntrinsicDriver:
    
    def __init__(self, config: config.Config):
        self._config = config
        self.case = None
        self.simulation = None
        self.sol = None
        self.set_simulation()
        self.set_sol()
        self.set_systems()
        
    def pre_simulation(self):

        if self._config.driver.compute_presimulation:
            self._compute_modalshapes()
            self._compute_modalcouplings()
        else:
            self._load_modalshapes()
            self._load_modalcouplings()

    def run_case(self):
        self.set_case()
        self.set_simulation()
        self.simulation.trigger()

    def set_case(self):
        # Configure the case
        pass
    
    def set_simulation(self):
        # Configure the simulation
        cls_simulation = fem4inas.simulations.factory_method(
            self._config.simulation.typeof)
        self.simulation = cls_simulation(self.sol,
                                         self._config.simulation)

    def set_sol(self):
        # Configure the simulation
        self.sol = solution.IntrinsicSolution(
            self._config.driver.solution_path)

    def set_systems(self):
        
    def _compute_modalshapes(self):
        
        modal_shapes = modes.shapes()
        self.sol.add_container('Modes', *modal_shapes)
        
    def _compute_modalcouplings(self):
        
        gammas = couplings.gammas()
        self.sol.add_container('Couplings', *gammas)

    def _load_modalshapes(self):

        self.sol.load_container('Modes')

    def _load_modalcouplings(self):
        self.sol.load_container('Couplings')
