from abc import ABC, abstractmethod

import feniax.preprocessor.solution as solution
import feniax.systems.system as system


__SIMULATION_DICT__ = dict()


class Simulation(ABC):
    
    def __init__(self, systems: dict[system.System], sol: solution.Solution, settings):
        """Manages how the various systems are run

        Parameters
        ----------
        systems : dict[system.System]
            System objects
        sol : solution.Solution
            Solution object
        settings : 
            Simulation settings

        """
        
        self.systems = systems
        self.sol = sol
        self.settings = settings

    @abstractmethod
    def trigger(self):
        """Launch the simulations
        
        """
        pass

    @abstractmethod
    def _run_systems(self):
        """Implements logic to run the systems: serial, parallel...

        """
        pass

    @abstractmethod
    def _post_run(self):
        """Anything to run after the systems are solved

        """
        pass

    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __SIMULATION_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __SIMULATION_DICT__[kwargs["cls_name"]] = cls
