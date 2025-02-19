from abc import ABC, abstractmethod

__FORAGER_DICT__ = dict()


class Forager(ABC):

    @abstractmethod
    def build_configs(self):
        pass

    @abstractmethod
    def spawn(self):
        pass
    
    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __FORAGER_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __FORAGER_DICT__[kwargs["cls_name"]] = cls
