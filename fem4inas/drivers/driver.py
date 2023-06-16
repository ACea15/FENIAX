from abc import ABC, abstractmethod

class Driver:

    @abstractmethod
    def set_simulation():
        ...

    @abstractmethod
    def run_simulation():
        ...

    @abstractmethod
    def close_simulation():
        ...

    @abstractmethod
    def get_simulation():
        ...

class DIntrinsicModal:

    def build():
        ...
        # calculate_modes
        # calculate_tensors

    def run():
        ...

    def postprocess():
        ...
    
