from abc import ABC, abstractmethod
import jax.numpy as jnp

class Registry:
    _registry = {}

    @classmethod
    def register(cls, key):
        def decorator(factory_class):
            cls._registry[key] = factory_class
            return factory_class
        return decorator

    @classmethod
    def create_instance(cls, key, *args, **kwargs):
        if key in cls._registry:
            factory_class = cls._registry[key]
            return factory_class(*args, **kwargs)
        else:
            raise KeyError(f"Class 'Gust{key}' not found in the registry")

class Shapes:

    @staticmethod
    def const(*args, **kwargs):
        return 1.

class Gust(ABC):

    @abstractmethod
    def calculate_downwash(self):
        """
        NpxNt
        """
        ...
    @abstractmethod
    def calculate_normals(self):
        ...
    @abstractmethod
    def set_solution(self, sol):
        ...

@Registry.register("Roger1Mc")
class GustRogerMc(Gust):

    def __init__(self,
                 

                 u_inf, xgust, gust_shift, simulation_time,
                 collocation_points, shape="const", dihedral_vector=None):

        self.u_inf = u_inf
        self.xgust = xgust
        self.gust_shift = gust_shift
        self.simulation_time = simulation_time
        self.collocation_points = collocation_points
        self._define_time()
        self._define_spanshape()
        self.calculate_downwash()
        
    def _define_time(self):

        self.gust_length = self.xgust[-1] - self.xgust[0]
        self.gust_totaltime = self.gust_length / self.u_inf
        time_discretization = (self.gust_shift + self.xgust) / self.u_inf
        if time_discretization[-1] < self.simulation_time[-1]:
            self.time = jnp.hstack([0., time_discretization, self.simulation_time[-1]])
        else:
            self.time = jnp.hstack([0., time_discretization])

    def calculate_downwash(self):
        """
        NpxNt panel downwash in time
        """
        #TODO: Optimize
        self.gust = jnp.zeros((self.npanels, self.ntime))
        self.gust_dot = jnp.zeros((self.npanels, self.ntime))
        self.gust_ddot = jnp.zeros((self.npanels, self.ntime))
        coeff = 2. * jnp.pi * self.u_inf / self.gust_length
        for panel in range(self.num_panels):
            delay=(self.collocation_points[panel,0]
                   - self.gust_position) / self.u_inf
            shape_span = self.shape_span(self.collocation_points[panel,1])
            filter_time = jnp.where((self.time >= delay) &
                                    (self.time <= delay +
                                     self.gust_totaltime), 1, 0)
            self.gust = self.gust.at[panel].set(shape_span * self.normals[panel] *
                                                    self.gust_intensity / (self.u_inf*2) *
                                                    (1 - jnp.cos(coeff * self.time))
                                                    )
            self.gust = self.gust * filter_time
            self.gust_dot = self.gust_dot.at[panel].set(shape_span * self.normals[panel] *
                                                            self.gust_intensity / (self.u_inf*2) *
                                                            (jnp.sin(coeff * self.time)) * coeff
                                                            )
            self.gust_dot = self.gust_dot * filter_time 
            self.gust_ddot = self.gust_ddot.at[panel].set(shape_span * self.normals[panel] *
                                                            self.gust_intensity / (self.u_inf*2) *
                                                            (jnp.sin(coeff * self.time)) * coeff**2
                                                            )
            self.gust_ddot = self.gust_ddot * filter_time
        self._define_eta()

    def _define_spanshape(self):
                                          
        self.shape_span = getattr(Shapes, self.shape)
                                          
    def calculate_normals(self):
                                          
        if self.dihedral is not None:
            self.normals = self.dihedral
    
    def set_solution(self, sol, sys_name):
        
        sol.add_container('GustRoger', label="_" + sys_name,
                          w=self.gust,
                          wdot=self.gust_dot,
                          wddot=self.gust_ddot,
                          x=self.time,
                          Qhj_w=self.Q_w,
                          Qhj_wdot=self.Q_wdot,
                          Qhj_wddot=self.Q_wddot,
                          Qhj_wsum=self.Q_wsum,
                          Qhjl_wdot=self.Ql
                          )

    def _define_eta(self):
        """
        NtxNm
        """

        
        self.Q_w = self.D0hat @ self.gust
        self.Q_wdot = self.D1hat @ self.gust_dot
        self.Q_wddot = self.D2hat @ self.gust_ddot
        self.Q_wsum = self.Q_w + self.Q_wdot + self.Q_wddot  # NmxNt
        self.Ql_wdot = jnp.tensordot(self.Dphat,
                                     self.gust_dot, axis=(1,0)) # NmxNtxNp
