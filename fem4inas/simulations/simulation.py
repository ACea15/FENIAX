from abc import ABC, abstractmethod

__SIMULATION_DICT__ = dict()

    
class Simulation(ABC):

    def __init__(self, systems, sol, settings):

        self.systems = systems
        self.sol = sol
        self.settings = settings
        
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

    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __SIMULATION_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __SIMULATION_DICT__[kwargs["cls_name"]] = cls

