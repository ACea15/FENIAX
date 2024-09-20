from abc import ABC, abstractmethod
import feniax.preprocessor.containers.data_container as data_container

__SYSTEM_DICT__ = dict()


class System(ABC):
    def __init__(self, name: str, settings: data_container.DataContainer):
        self.name = name
        self.settings = settings

    # @abstractmethod
    # def set_init(self):
    #     pass

    # @abstractmethod
    # def set_name(self):
    #     pass

    @abstractmethod
    def set_system(self):
        pass

    @abstractmethod
    def set_solver(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def save(self):
        pass

    # @abstractmethod
    # def pull_solution(self):
    #     pass

    def __init_subclass__(cls, **kwargs):
        assert "cls_name" in kwargs
        super().__init_subclass__()
        if kwargs["cls_name"] in __SYSTEM_DICT__:
            raise ValueError("Name %s already registered!" % kwargs["cls_name"])
        __SYSTEM_DICT__[kwargs["cls_name"]] = cls
