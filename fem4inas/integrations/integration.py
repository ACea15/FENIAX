from abc import ABC, abstractmethod

import fem4inas.preprocessor.inputs.Inputs as Inputs
import fem4inas.intrinsic.modes as modes

class Integration(ABC):
    @abstractmethod
    def run(self):
        pass


class IntrinsicIntegration(Integration):
    
    def __init__(self, config: Inputs):

        self.config = config

    def run(self):
        pass

    def compute_modalshapes(self):
        # Implement compute_modalshapes for IntrinsicIntegration
        pass

    def compute_modalcouplings(self):
        # Implement compute_modalcouplings for IntrinsicIntegration
        pass
