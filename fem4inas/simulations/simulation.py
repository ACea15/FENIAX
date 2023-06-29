from abc import ABC, abstractmethod

__SIMULATION_DICT__ = dict()

    
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

    def __init_subclass__(cls, **kwargs):
        assert "name" in kwargs
        super().__init_subclass__()
        if kwargs["name"] in __SIMULATION_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["name"])
        __SIMULATION_DICT__[kwargs["name"]] = cls

