from abc import ABC, abstractmethod
import jax.numpy as jnp
from fem4inas.intrinsic.utils import Registry
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsicmodal
import fem4inas.preprocessor.solution as solution

class Shapes:

    @staticmethod
    def const(*args, **kwargs):
        return 1.

class Gust(ABC):
    @abstractmethod
    def calculate_normals(self):
        ...

    @abstractmethod
    def calculate_downwash(self):
        """
        NpxNt
        """
        ...
        
    @abstractmethod
    def set_solution(self, sol):
        ...

@Registry.register("GustRogerMc")
class GustRogerMc(Gust):

    def __init__(self,
                 settings: intrinsicmodal.Dsystem,
                 sol: solution.IntrinsicSolution):

        self.settings = settings
        self.solaero = getattr(sol.data,
                               f"modalaeroroger_{settings.name}")
        self.gust = None
        self.gust_dot = None
        self.gust_ddot = None
        self.shape_span = None
        self.time = None
        self.gust_length = None
        self.collocation_points = None
        self.gust_totaltime = None
        self.ntime = None
        self.npanels = None
        self.u_inf = None
        self.rho_inf = None
        self._set_flow()
        self._set_gust()
        
    def _set_flow(self):
        self.u_inf = self.settings.aero.u_inf
        self.rho_inf = self.settings.aero.rho_inf
        
    def _set_gust(self):
        
        self.gust_step = self.settings.aero.gust.step
        self.gust_shift = self.settings.aero.gust.shift
        simulation_time = self.settings.t
        self.collocation_points = self.settings.aero.gust.collocation_points
        self.dihedral = self.settings.aero.gust.panels_dihedral
        self.gust_length = self.settings.aero.gust.length
        self.gust_intensity = self.settings.aero.gust.intensity
        self.gust_totaltime = self.gust_length / self.u_inf
        self.xgust = jnp.arange(jnp.min(self.collocation_points[:,0]),
                                jnp.max(self.collocation_points[:,0]) +
                                self.gust_length + self.gust_step,
                                self.gust_step)
        time_discretization = (self.gust_shift + self.xgust) / self.u_inf
        if time_discretization[-1] < simulation_time[-1]:
            self.time = jnp.hstack([time_discretization,
                                    time_discretization[-1] + 1e-6,
                                    simulation_time[-1]])
        else:
            self.time = time_discretization
        if self.time[0] != 0.:
            self.time = jnp.hstack([0.,
                                    self.time[0] - 1e-6,
                                    self.time])
        self.ntime = len(self.time)
        self.npanels = len(self.collocation_points)
        self._define_spanshape(self.settings.aero.gust.shape)
        
    def _define_spanshape(self, shape):

        self.shape_span = getattr(Shapes, shape)

    def calculate_downwash(self):
        """
        NpxNt panel downwash in time
        """
        #TODO: Optimize
        self.gust = jnp.zeros((self.npanels, self.ntime))
        self.gust_dot = jnp.zeros((self.npanels, self.ntime))
        self.gust_ddot = jnp.zeros((self.npanels, self.ntime))
        coeff = 2. * jnp.pi * self.u_inf / self.gust_length
        for panel in range(self.npanels):
            delay = (self.collocation_points[panel, 0]
                     + self.gust_shift) / self.u_inf
            shape_span = self.shape_span(self.collocation_points[panel,1])
            filter_time = jnp.where((self.time >= delay) &
                                    (self.time <= delay +
                                     self.gust_totaltime), 1, 0)
            self.gust = self.gust.at[panel].set(
                filter_time * (shape_span * self.normals[panel] *
                               self.gust_intensity / (self.u_inf*2) *
                               (1 - jnp.cos(coeff * (self.time - delay)))
                               ))
            self.gust_dot = self.gust_dot.at[panel].set(
                filter_time * (shape_span * self.normals[panel] *
                               self.gust_intensity / (self.u_inf*2) *
                               (jnp.sin(coeff * (self.time - delay))) * coeff
                               ))
            self.gust_ddot = self.gust_ddot.at[panel].set(
                filter_time * (shape_span * self.normals[panel] *
                               self.gust_intensity / (self.u_inf*2) *
                               (jnp.cos(coeff * (self.time - delay))) * coeff**2
                               ))
        self._define_eta()

    def calculate_normals(self):

        if self.dihedral is not None:
            self.normals = self.dihedral
    
    def set_solution(self, sol: solution.IntrinsicSolution,
                     sys_name: str):
        
        sol.add_container('GustRoger', label="_" + sys_name,
                          w=self.gust,
                          wdot=self.gust_dot,
                          wddot=self.gust_ddot,
                          x=self.time,
                          Qhj_w=self.Q_w,
                          Qhj_wdot=self.Q_wdot,
                          Qhj_wddot=self.Q_wddot,
                          Qhj_wsum=self.Q_wsum,
                          Qhjl_wdot=self.Ql_wdot
                          )

    def _define_eta(self):
        """
        NtxNm
        """
        D0hat = self.solaero.D0hat  # NbxNm
        D1hat = self.solaero.D1hat
        D2hat = self.solaero.D2hat
        D3hat = self.solaero.D3hat
        self.Q_w = D0hat @ self.gust
        self.Q_wdot = D1hat @ self.gust_dot
        self.Q_wddot = D2hat @ self.gust_ddot
        self.Q_wsum = self.Q_w + self.Q_wdot + self.Q_wddot  # NmxNt
        self.Ql_wdot = jnp.tensordot(D3hat,  # NpxNmxNb
                                     self.gust_dot,  # NbxNt
                                     axes=(2,0))  # NpxNmxNt
