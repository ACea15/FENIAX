import itertools

import feniax.intrinsic.dynamicShard as dynamic_shard
import feniax.intrinsic.staticFast as staticFast
import feniax.intrinsic.dynamicFast as dynamicFast
import feniax.intrinsic.xloads as xloads
from feniax.systems.intrinsic_system import IntrinsicSystem

import logging
import jax
from functools import partial
from feniax.ulogger.setup import get_logger

logger = get_logger(__name__)

class IntrinsicFastSystem(IntrinsicSystem, cls_name="Fast_intrinsic"):

    def set_xloading(self):
        """
        Setting of loads happen internally in the main function
        """
        pass

    def set_solver(self):
        """
        It happens internally
        """
        pass

    def set_args(self):
        """
        Solution doesn't use args as they are picked inside the function
        """
        
        pass
    
    def solve(self):
        
        results = self.main(q0=self.q0,
                            config=self.config
                            )
        
        self.build_solution(**results)

    def build_solution(self, **kwargs):
        
        self.sol.add_container("Couplings",
                               gamma1=kwargs.get("gamma1"),
                               gamma2=kwargs.get("gamma2"))
        self.sol.add_container("Modes",
                               phi1=kwargs.get("phi1"),
                               psi1=kwargs.get("psi1"),
                               phi2=kwargs.get("phi2"),
                               phi1l=kwargs.get("phi1l"),
                               phi1ml=kwargs.get("phi1ml"),
                               psi1l=kwargs.get("psi1l"),
                               phi2l=kwargs.get("phi2l"),
                               psi2l=kwargs.get("psi2l"),
                               omega=kwargs.get("omega"),
                               X_xdelta=kwargs.get("X_xdelta"),
                               C0ab=kwargs.get("C0ab"),
                               C06ab=kwargs.get("C06a")
                               )
        if self.settings.save:
            self.sol.save_container("Modes")
            self.sol.save_container("Couplings")
        
class StaticFastIntrinsic(IntrinsicFastSystem, cls_name="staticFast_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        self.label = f"main_{label_sys}"
        logger.debug(f"Setting {self.__class__.__name__} with label {self.label}")
        self.main = partial(jax.jit, static_argnames=["config"])(
            getattr(staticFast, self.label))

    def build_solution(self, q, X2, X3, ra, Cab, *args, **kwargs):
        
        super().build_solution(**kwargs)
        self.sol.add_container(
            "StaticSystem",
            label="_" + self.name,
            q=q,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
            t=self.settings.t,
        )
        if self.settings.save:
            self.sol.save_container("StaticSystem", label="_" + self.name)

class DynamicFastIntrinsic(IntrinsicFastSystem, cls_name="dynamicFast_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        self.label = f"main_{label_sys}"
        logger.debug(f"Setting intrinsic Dynamic Fast system with label {self.label}")
        self.main = partial(jax.jit, static_argnames=["config"])(
            getattr(dynamicFast, self.label))

    def build_solution(self, q, X1, X2, X3, ra, Cab, **kwargs):

        super().build_solution(**kwargs)
        self.sol.add_container(
            "DynamicSystem",
            label="_" + self.name,
            q=q,
            X1=X1,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
            t=self.settings.t,
        )
        if self.settings.save:
            self.sol.save_container("DynamicSystem", label="_" + self.name)
        
