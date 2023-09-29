from abc import ABC, abstractmethod
import fem4inas.drivers.driver as driver
import fem4inas.preprocessor.solution as solution
import fem4inas.systems.system as system
__SIMULATION_DICT__ = dict()

    
class Simulation(ABC):

    def __init__(self, systems: dict[system.System],
                 sol: solution.Solution,
                 settings):

        self.systems = systems
        self.sol = sol
        self.settings = settings
        
    @abstractmethod
    def trigger(self):
        pass

    @abstractmethod
    def _run_systems(self):
        pass

    @abstractmethod
    def _post_run(self):
        pass

    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __SIMULATION_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __SIMULATION_DICT__[kwargs["cls_name"]] = cls

