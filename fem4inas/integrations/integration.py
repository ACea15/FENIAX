from abc import ABC, abstractmethod

class Integration(ABC):
    @abstractmethod
    def run(self):
        pass

class IntrinsicIntegration(Integration):

    def run(self):
        pass

    def compute_modalshapes(self):
        # Implement compute_modalshapes for IntrinsicIntegration
        pass

    def compute_modalcouplings(self):
        # Implement compute_modalcouplings for IntrinsicIntegration
        pass
