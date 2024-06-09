import pathlib
from abc import ABC, abstractmethod
import jax.numpy as jnp
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsicmodal
import fem4inas.preprocessor.solution as solution
from fem4inas.intrinsic.utils import Registry

import copy


class ModalAero(ABC):

    @abstractmethod
    def set_container():
        ...

    @abstractmethod
    def get_matrices():
        ...

    @abstractmethod
    def save_sol():
        ...
        
@Registry.register("AeroRoger")
class AeroRoger(ModalAero):

    def __init__(self,
                 system: intrinsicmodal.Dsystem,
                 sol: solution.IntrinsicSolution):
        self.sys = system
        self.settings = system.aero
        self.sol = sol
        self.container = None
        
    def set_container(self, container):
        self.container = copy.deepcopy(container)
        
    def save_sol(self):
        self.sol.add_container("ModalAeroRoger",
                               label="_"+self.sys.name,
                               **self.container)
    
    def _set_flow(self):
        self.u_inf = self.sys.aero.u_inf
        self.rho_inf = self.sys.aero.rho_inf
        self.c_ref = self.sys.aero.c_ref
        self.q_inf = self.sys.aero.q_inf

    def _build_rfa(self):
        ...
        
    def _get_matrix(self, matrix: jnp.array, name: str):
        
        if len(matrix.shape) == 2:
            self.container[f"{name}0"] = matrix
        elif len(matrix.shape) == 3 and len(matrix) > 4:
            self.container[f"{name}0"] = matrix[0]
            self.container[f"{name}1"] = matrix[1]
            self.container[f"{name}2"] = matrix[2]
            self.container[f"{name}3"] = matrix[3:]
        else:
            pass # TODO: assert error
    def get_matrices(self, scale=True):
        if self.container is None:
            self._build_matrices()
        if scale:
            self._scale()
            
    def _build_matrices(self):
        
        self.container = dict()
        if self.settings.poles is not None:
            self.container.update(poles=self.settings.poles)
        # GAFs structure
        if self.settings.Qk_struct is not None:
            if len(self.settings.Qk_struct[0]) == 1: # steady
                A0 = self.settings.Qk_struct[1]
                self.container.update(A0=A0)
            else:
                ... # build rfa
        elif self.settings.A is not None:
            self._get_matrix(self.settings.A, "A")
        # GAFs gust
        if self.settings.Qk_gust is not None:
            # build rfa
            ...
        elif self.settings.D is not None:
            self._get_matrix(self.settings.D, "D")
        # GAFs controls
        if self.settings.Qk_controls is not None:
            if len(self.settings.Qk_controls[0]) == 1: # steady
                B0 = self.settings.Qk_struct[1]
                self.container.update(B0=B0)
            else:
                ... # build rfa
        elif self.settings.B is not None:
            self._get_matrix(self.settings.B, "B")
        # GAFs steady
        if self.settings.Q0_rigid is not None:
            C0 = self.settings.Q0_rigid
            self.container.update(C0=C0)
            
    def _scale(self):

        self._set_flow()
        container_entries = list(self.container.keys())
        for k in container_entries:
            v = self.container[k]
            try:
                if int(k[-1]) == 0:
                    self.container[f"{k}hat"] = self.q_inf * v
                elif int(k[-1]) == 1:
                    self.container[f"{k}hat"] = (self.c_ref * self.rho_inf *
                                                 self.u_inf / 4 * v)
                elif int(k[-1]) == 2:
                    self.container[f"{k}hat"] = (self.c_ref**2 * self.rho_inf /
                                                  8 * v)
                elif int(k[-1]) == 3:
                    self.container[f"{k}hat"] = self.q_inf * v
            except ValueError:
                continue
        if "A2hat" in self.container.keys():
            A2hat = (jnp.eye(len(self.container["A2hat"])) -
                     self.container["A2hat"])
            self.container["A2hatinv"] = jnp.linalg.inv(A2hat)
