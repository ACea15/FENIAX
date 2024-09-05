from abc import ABC, abstractmethod

__DRIVER_DICT__ = dict()


class Driver(ABC):
    def run_cases(self):
        self.set_case()
        self.set_integration()
        self.set_simulation()
        # Perform other operations using the configured case, integration, and simulation

    def set_case(self):
        # Configure the case
        pass

    def set_integration(self):
        # Configure the integration
        pass

    def set_simulation(self):
        # Configure the simulation
        pass

    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __DRIVER_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __DRIVER_DICT__[kwargs["cls_name"]] = cls
