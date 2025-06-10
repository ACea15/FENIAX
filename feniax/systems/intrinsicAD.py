from feniax.systems.system import System
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.staticAD as static_ad
import feniax.intrinsic.dynamicAD as dynamic_ad
import feniax.intrinsic.postprocess as postprocess
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
import feniax.intrinsic.initcond as initcond
import feniax.intrinsic.args as libargs
import feniax.intrinsic.objectives as objectives
from functools import partial
import jax
import jax.numpy as jnp
import logging
from feniax.ulogger.setup import get_logger

logger = get_logger(__name__)

class IntrinsicADSystem(System, cls_name="AD_intrinsic"):
    def __init__(
        self,
        name: str,
        settings: intrinsicmodal.Dsystem,
        fem: intrinsicmodal.Dfem,
        sol: solution.IntrinsicSolution,
        config,
    ):
        self.name = name
        self.settings = settings
        self.fem = fem
        self.sol = sol
        self.config = config
        # self._set_xloading()
        # self._set_generator()
        # self._set_solver()

    def set_ic(self, q0):
        if q0 is None:
            self.q0 = jnp.zeros(self.settings.num_states)
        else:
            self.q0 = q0

    def set_solver(self):
        match self.settings.ad.grad_type:
            case "grad":
                raise ValueError(
                    "Dropped support for grad as it requires value-function output which prevents generalisations"
                )
                # self.eqsolver = jax.grad
                # obj_label = f"{self.settings.ad.objective_var}_{self.settings.ad.objective_fun.upper()}grad"
                # self.f_obj = objectives.factory(obj_label)

            case "value_grad":
                raise ValueError(
                    "Dropped support for grad as it requires value-function output which prevents generalisations"
                )
                # self.eqsolver = jax.value_and_grad
                # self.eqsolver = jax.grad
                # obj_label = f"{self.settings.ad.objective_var}_{self.settings.ad.objective_fun.upper()}grad"
                # self.f_obj = objectives.factory(obj_label)
            case "jacrev":
                self.eqsolver = jax.jacrev
                obj_label = f"{self.settings.ad.objective_var}_{self.settings.ad.objective_fun.upper()}"
                self.f_obj = objectives.factory(obj_label)
            case "jacfwd":
                self.eqsolver = jax.jacfwd
                obj_label = f"{self.settings.ad.objective_var}_{self.settings.ad.objective_fun.upper()}"
                self.f_obj = objectives.factory(obj_label)
            case "value":
                self.eqsolver = lambda func, *args, **kwargs: func
                obj_label = f"{self.settings.ad.objective_var}_{self.settings.ad.objective_fun.upper()}"
                self.f_obj = objectives.factory(obj_label)
            case _:
                raise ValueError(f"Incorrect solver {self.settings.ad.grad_type}")

    def solve(self, static_argnames=["config", "f_obj", "obj_args"], *args, **kwargs):
        
        logger.info(f"Running System solution")
        
        # TODO: option to jit at the end, jit dFq, or not jit at all.
        if True: # not working with diffrax static solver
            fprime = partial(jax.jit, static_argnames=static_argnames)(self.eqsolver(self.dFq, has_aux=True))  # call to jax.grad..etc
        elif False:
            fprime = (self.eqsolver(partial(jax.jit,
                                            static_argnames=static_argnames)(self.dFq),
                                    has_aux=True))  # call to jax.grad..etc
        else:
            fprime = self.eqsolver(self.dFq, has_aux=True)
        if self.settings.ad.grad_type == "value_grad":
            ((val, fout), jac) = fprime(
                self.settings.ad.inputs,
                q0=self.q0,
                config=self.config,
                f_obj=self.f_obj,
                obj_args=self.settings.ad.objective_args,
                *args,
                **kwargs
            )
        else:
            jac, fout = fprime(
                self.settings.ad.inputs,
                q0=self.q0,
                config=self.config,
                f_obj=self.f_obj,
                obj_args=self.settings.ad.objective_args,
                *args,
                **kwargs                
            )
        self.build_solution(jac, *fout)

    def save(self):
        pass

    def set_args(self):
        pass
    
    def set_eta0(self):
        pass

    def set_states(self):
        self.settings.build_states(self.fem.num_modes, self.fem.num_nodes)

    def set_xloading(self):
        pass


class StaticADIntrinsic(IntrinsicADSystem, cls_name="staticAD_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        label_ad = self.settings.ad.label
        label = f"main_{label_sys}_{label_ad}"
        logger.debug(f"Setting {self.__class__.__name__} with label {label}")
        self.dFq = getattr(static_ad, label)

    def build_solution(self, jac, objective, q, X1, X2, X3, ra, Cab, *args, **kwargs):
        self.sol.add_container(
            "StaticSystem",
            label="_" + self.name,
            jac=jac,
            q=q,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
            f_ad=objective,
        )
        if self.settings.save:
            self.sol.save_container("StaticSystem", label="_" + self.name)


class DynamicADIntrinsic(IntrinsicADSystem, cls_name="dynamicAD_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        label_ad = self.settings.ad.label
        label = f"main_{label_sys}_{label_ad}"
        logger.debug(f"Setting {self.__class__.__name__} with label {label}")
        self.dFq = getattr(dynamic_ad, label)

    def build_solution(
        self, jac, objective, q, X1, X2, X3, ra, Cab, *args, f_ad=None, **kwargs
    ):
        self.sol.add_container(
            "DynamicSystem",
            label="_" + self.name,
            jac=jac,
            q=q,
            X1=X1,
            X2=X2,
            X3=X3,
            t=self.settings.t,
            Cab=Cab,
            ra=ra,
            f_ad=objective
        )
        if self.settings.save:
            self.sol.save_container("DynamicSystem", label="_" + self.name)
