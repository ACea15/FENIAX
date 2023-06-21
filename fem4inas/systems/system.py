from abc import ABC, abstractmethod

class System(ABC):
    @abstractmethod
    def set_init(self):
        pass

    @abstractmethod
    def set_name(self):
        pass

    @abstractmethod
    def set_generator(self):
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
