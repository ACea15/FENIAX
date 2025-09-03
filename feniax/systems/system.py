from abc import ABC, abstractmethod
import feniax.preprocessor.containers.data_container as data_container

__SYSTEM_DICT__ = dict()


class System(ABC):
    
    def __init__(self, name: str, settings: data_container.DataContainer):
        """Initialise object representing system of equations

        Parameters
        ----------
        name : str
            Name of the system
        settings : data_container.DataContainer
            system configuration settings

        """
        
        self.name = name
        self.settings = settings

    @abstractmethod
    def set_ic(self):
        """
        Initial conditions for the system
        """
        pass
        
    @abstractmethod
    def set_system(self):
        """
        Defines the actual system to be solved based on the inputs
        """
        pass

    @abstractmethod
    def set_solver(self):
        """
        Picks the solver in ./sollibs to solve the system of equations
        """
        
        pass

    @abstractmethod
    def solve(self):
        """
        Run the solver on the system of equations
        """
        
        pass

    @abstractmethod
    def build_solution(self):
        """
        Based on the solution states, build any postprecessing fields
        """

    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __SYSTEM_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __SYSTEM_DICT__[kwargs["cls_name"]] = cls
