import fem4inas.intrinsic.modes as modes
import fem4inas.intrinsic.couplings as couplings
from fem4inas.drivers.driver import Driver

import fem4inas.simulations
from fem4inas.preprocessor import solution, configuration

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
    
    def __init__(self, config: configuration.Config):
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
        self.num_systems = 0
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

        if self.num_systems == 0:
            print("no systems in the simulation")
        else:
            self.simulation.trigger()
    
    def _set_simulation(self):
        # Configure the simulation
        if hasattr(self._config, "simulation"):
            cls_simulation = fem4inas.simulations.factory(
                self._config.simulation.typeof)
            self.simulation = cls_simulation(self.systems,
                                             self.sol,
                                             self._config.simulation)
        else:
            print("no simulation settings")
            
    def _set_systems(self):

        self.systems = dict()
        if hasattr(self._config, "systems"):
            for k, v in self._config.systems.sys.items():
                cls_sys = fem4inas.systems.factory(
                    v.typeof)
                self.systems[k] = cls_sys(k, v,
                                          self._config.fem,
                                          self.sol)
        self.num_systems = len(self.systems)

    def _set_sol(self):
        # Configure the simulation
        self.sol = solution.IntrinsicSolution(
            self._config.driver.solution_path)
        
    def _compute_modalshapes(self):
        
        modal_shapes = modes.shapes(self._config.fem.X,
                                    self._config.fem.Ka,
                                    self._config.fem.Ma,
                                    self._config)
        self.sol.add_container('Modes', *modal_shapes)
        
    def _compute_modalcouplings(self):
        
        gamma1 = couplings.f_Gamma1(self.sol.modes.phi1,
                                    self.sol.modes.psi1)
        gamma2 = couplings.f_Gamma2(self.sol.modes.phi1ml,
                                    self.sol.modes.phi2,
                                    self.sol.modes.psi2,
                                    self.sol.modes.X_xdelta)

        self.sol.add_container('Couplings', gamma1, gamma2)

    def _load_modalshapes(self):

        self.sol.load_container('Modes')

    def _load_modalcouplings(self):
        self.sol.load_container('Couplings')
