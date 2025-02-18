from abc import ABC, abstractmethod

__DRIVER_DICT__ = dict()


class Driver(ABC):

    @abstractmethod
    def run_cases(self):
        pass
    
    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __DRIVER_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __DRIVER_DICT__[kwargs["cls_name"]] = cls
