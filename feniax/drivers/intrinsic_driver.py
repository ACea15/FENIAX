
import feniax.intrinsic.galerkinmodes as galerkinmodes
import feniax.simulations
import feniax.systems
from feniax.drivers.driver import Driver
from feniax.preprocessor import configuration, solution
from feniax.ulogger.setup import get_logger
import feniax.foragers

logger = get_logger(__name__)

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
        self._set_sol()
        self._set_systems()
        self._set_simulation()

    def pre_simulation(self):
        # TODO: here the RFA for aerodynamics should be included
        # TODO: condensation methods of K and M should be included
        galerkin = galerkinmodes.Galerkin(self._config, self.sol)
        galerkin.compute()

    def run_cases(self):
        if self.num_systems == 0:
            logger.warning("no systems in the simulation")
        else:
            self.simulation.trigger()

    def post_simulation(self) -> None:
        
        if hasattr(self._config, "forager"):
            cls_forager = feniax.foragers.factory(
                f"forager_{self._config.forager.typeof}")
            forager = cls_forager(self._config,
                                  self.sol,
                                  self.systems)
            forager.build_configs()
            forager.spawn()
            
    def _set_simulation(self):
        # Configure the simulation
        if hasattr(self._config, "simulation"):
            logger.info(f"Initialising Simulation {self._config.simulation.typeof}")
            cls_simulation = feniax.simulations.factory(self._config.simulation.typeof)
            self.simulation = cls_simulation(
                self.systems, self.sol, self._config.simulation
            )
            
        else:
            logger.error("no simulation settings")

    def _set_systems(self):
        logger.info("Setting systems")
        self.systems = dict()
        if hasattr(self._config, "systems"):
            for k, v in self._config.systems.mapper.items():
                logger.info(f"Initialising system {k}")
                cls_sys = feniax.systems.factory(f"{v.solution}{v.operationalmode}_intrinsic")
                self.systems[k] = cls_sys(
                    k, v, self._config.fem, self.sol, self._config
                )
                logger.info(f"Initialised {v.solution}{v.operationalmode}_intrinsic")
        elif hasattr(self._config, "system"):
            name = self._config.system.name
            logger.info(f"Initialising system {name}")
            cls_sys = feniax.systems.factory(
                f"{self._config.system.solution}{self._config.system.operationalmode}_intrinsic"
            )
            self.systems[name] = cls_sys(
                name, self._config.system, self._config.fem, self.sol, self._config
            )
            logger.info(f"Initialised {self._config.system.solution}{self._config.system.operationalmode}_intrinsic")

        self.num_systems = len(self.systems)

    def _set_sol(self):
        # Configure the solution object
        self.sol = solution.IntrinsicSolution(self._config.driver.sol_path)

class IntrinsicMPIDriver(Driver, cls_name="intrinsicMPI"):
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

    def __init__(self, configs: list[configuration.Config]):
        """

        Parameters
        ----------
        config : config.Config
            Configuration object

        """
        self._configs = configs
        self.simulation = None
        self.sol = None
        self.systems = None
        self.num_systems = 0
        self._set_sol()
        self._set_systems()
        self._set_simulation()

    def pre_simulation(self):
        # TODO: here the RFA for aerodynamics should be included
        # TODO: condensation methods of K and M should be included
        
        galerkin = galerkinmodes.Galerkin(self._configs, self.sol)
        galerkin.compute()

    def run_cases(self):
        if self.num_systems == 0:
            logger.warning("no systems in the simulation")
        else:
            self.simulation.trigger()

    def post_simulation(self) -> None:
        # gather all solution objects
        # pass configs and sols to the forager to decide what combination of them
        # is the worse and launch the simulations
        if hasattr(self._configs, "forager"):
            cls_forager = feniax.foragers.factory(
                f"intrinsic_{self._configs.forager.typeof}")
            forager = cls_forager(self._configs,
                                  self.sol,
                                  self.systems)
            forager.build_configs()
            forager.spawn()
            
    def _set_simulation(self):
        # Configure the simulation
        if hasattr(self._configs, "simulation"):
            logger.info(f"Initialising Simulation {self._configs.simulation.typeof}")
            cls_simulation = feniax.simulations.factory(self._configs.simulation.typeof)
            self.simulation = cls_simulation(
                self.systems, self.sol, self._configs.simulation
            )
            
        else:
            logger.error("no simulation settings")

    def _set_systems(self):
        logger.info("Setting systems")
        self.systems = dict()
        if hasattr(self._configs, "systems"):
            for k, v in self._configs.systems.mapper.items():
                logger.info(f"Initialising system {k}")
                cls_sys = feniax.systems.factory(f"{v.solution}{v.operationalmode}_intrinsic")
                self.systems[k] = cls_sys(
                    k, v, self._configs.fem, self.sol, self._configs
                )
                logger.info(f"Initialised {v.solution}{v.operationalmode}_intrinsic")
        elif hasattr(self._configs, "system"):
            name = self._configs.system.name
            logger.info(f"Initialising system {name}")
            cls_sys = feniax.systems.factory(
                f"{self._configs.system.solution}{self._configs.system.operationalmode}_intrinsic"
            )
            self.systems[name] = cls_sys(
                name, self._configs.system, self._configs.fem, self.sol, self._configs
            )
            logger.info(f"Initialised {self._configs.system.solution}{self._configs.system.operationalmode}_intrinsic")

        self.num_systems = len(self.systems)

    def _set_sol(self):
        # Configure the solution object
        self.sol = solution.IntrinsicSolution(self._configs.driver.sol_path)
        
