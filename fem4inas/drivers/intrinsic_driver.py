import fem4inas.intrinsic.modes as modes
import fem4inas.intrinsic.couplings as couplings
from fem4inas.drivers.driver import Driver

import fem4inas.simulations
from fem4inas.preprocessor import solution, config

class IntrinsicDriver(Driver, cls_name="intrinsic"):
    """Driver for the modal intrinsic theory 

    Creates simulation, systems and solution data objects and calls
    the pre-simulation and the simulation public methods

    Parameters
    ----------
    config : config.Config
         Configuration object

    Attributes
    ----------
    _config : see Parameters
    simulation :
    sol :
    systems : 
    
    """
    
    def __init__(self, config: config.Config):
        """

        Parameters
        ----------
        config : config.Config
            Configuration object

        """
        
        self._config = config
        self.simulation = None
        self.sol = None
        self.systems = None
        self._set_systems()
        self._set_simulation()
        self._set_sol()
        
    def pre_simulation(self):

        #TODO: here the RFA for aerodynamics should be included
        #TODO: condensation methods of K and M should be included
        if self._config.driver.compute_presimulation:
            self._compute_modalshapes()
            self._compute_modalcouplings()
        else:
            self._load_modalshapes()
            self._load_modalcouplings()

    def run_case(self):
        
        self.simulation.trigger()
    
    def _set_simulation(self):
        # Configure the simulation
        cls_simulation = fem4inas.simulations.factory(
            self._config.simulation.typeof)
        self.simulation = cls_simulation(self.systems,
            self.sol,
            self._config.simulation)

    def _set_systems(self):

        self.systems = dict()
        for k, v in self._config.systems.data.items():
            cls_sys = fem4inas.systems.factory(
                v.typeof)
        self.systems[k] = cls_sys(k, v)

    def _set_sol(self):
        # Configure the simulation
        self.sol = solution.IntrinsicSolution(
            self._config.driver.solution_path)
        
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
