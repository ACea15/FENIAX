from abc import ABC, abstractmethod

__DRIVER_DICT__ = dict()


class Driver(ABC):
    """ Initialises the main objects of the program and triggers the simulation

    """
    @abstractmethod
    def _set_simulation(self):
        """Initialise the simulation object
        
        """
        pass
    
    @abstractmethod
    def _set_systems(self):
        """Initialise the system of equations objects
        
        """
        pass
    
    @abstractmethod
    def pre_simulation(self):
        """Run any computation pre-simulation

        """
        
        pass
    
    @abstractmethod
    def run_cases(self):
        """Trigger the simulations
        
        """
        pass

    @abstractmethod
    def post_simulation(self):
        """Run any computation post-simulation

        For instance the forager pattern that launches other programs

        """
        
        pass
    
    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __DRIVER_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __DRIVER_DICT__[kwargs["cls_name"]] = cls
