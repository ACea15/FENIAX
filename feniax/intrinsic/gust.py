from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
from feniax.intrinsic.utils import Registry
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
from functools import partial


class Shapes:
    @staticmethod
    def const(*args, **kwargs):
        return 1.0


class Gust(ABC):
    @abstractmethod
    def calculate_normals(self): ...

    @abstractmethod
    def calculate_downwash(self):
        """
        NpxNt
        """
        ...

    @abstractmethod
    def set_solution(self, sol): ...


def _get_spanshape(shape):
    shape_span = getattr(Shapes, shape)
    return shape_span


# @partial(jax.jit, static_argnames=['min_collocationpoints', 'max_collocationpoints'])
# @jax.jit
# def _get_gustRogerMc(
#     gust_intensity,
#     dihedral,
#     gust_shift,
#     gust_step,
#     simulation_time,
#     gust_length,
#     u_inf,
#     min_collocationpoints,
#     max_collocationpoints,
# ):
#     #
#     gust_totaltime = gust_length / u_inf
#     xgust = jnp.arange(
#         min_collocationpoints,  # jnp.min(collocation_points[:,0]),
#         max_collocationpoints  # jnp.max(collocation_points[:,0]) +
#         + gust_length
#         + gust_step,
#         gust_step,
#     )
#     time_discretization = (gust_shift + xgust) / u_inf
#     # if time_discretization[-1] < simulation_time[-1]:
#     #     time = jnp.hstack(
#     #         [time_discretization, time_discretization[-1] + 1e-6, simulation_time[-1]]
#     #     )
#     # else:
#     #     time = time_discretization
#     extended_time = jnp.hstack(
#         [time_discretization, time_discretization[-1] + 1e-6, simulation_time[-1]]
#     )
#     time = jax.lax.select(time_discretization[-1] < simulation_time[-1],
#                           extended_time,
#                           jnp.hstack([time_discretization,
#                                       time_discretization[-1] + 1e-6,
#                                       time_discretization[-1] + 2*1e-6]) # need to be shame shape!!
#                           )
    
#     # if time[0] != 0.0:
#     #     time = jnp.hstack([0.0, time[0] - 1e-6, time])
#     time = jax.lax.select(time[0] != 0.0,
#                           jnp.hstack([0.0, time[0] - 1e-6, time]),
#                           jnp.hstack([0.0, 1e-6, 2*1e-6, time[1:]]))
        
#     ntime = len(time)
#     # npanels = len(collocation_points)
#     return gust_totaltime, xgust, time, ntime


@partial(jax.jit, static_argnames=["fshape_span"])
def _downwashRogerMc(
    u_inf,
    gust_length,
    gust_intensity,
    gust_shift,
    collocation_points,
    normals,
    time,
    gust_totaltime,
    fshape_span
):
    """
    NpxNt panel downwash in time
    """
    coeff = 2.0 * jnp.pi * u_inf / gust_length

    @jax.jit
    def kernel(collocation_point, normal):
        delay = (collocation_point[0] + gust_shift) / u_inf
        shape_span = fshape_span(collocation_point[1])
        filter_time = jnp.where(
            (time >= delay) & (time <= delay + gust_totaltime), 1, 0
        )

        gust = filter_time * (
            shape_span
            * normal
            * gust_intensity
            / (u_inf * 2)
            * (1 - jnp.cos(coeff * (time - delay)))
        )
        gust_dot = filter_time * (
            shape_span
            * normal
            * gust_intensity
            / (u_inf * 2)
            * (jnp.sin(coeff * (time - delay)))
            * coeff
        )
        gust_ddot = filter_time * (
            shape_span
            * normal
            * gust_intensity
            / (u_inf * 2)
            * (jnp.cos(coeff * (time - delay)))
            * coeff**2
        )
        return gust, gust_dot, gust_ddot

    f1 = jax.vmap(kernel, in_axes=(0, 0), out_axes=(0, 0, 0))
    gust, gust_dot, gust_ddot = f1(collocation_points, normals)
    return gust, gust_dot, gust_ddot


@jax.jit
def _getGAFs(
    D0hat,  # NbxNm
    D1hat,
    D2hat,
    D3hat,
    gust,
    gust_dot,
    gust_ddot,
):
    Q_w = D0hat @ gust
    Q_wdot = D1hat @ gust_dot
    Q_wddot = D2hat @ gust_ddot
    Q_wsum = Q_w + Q_wdot + Q_wddot  # NmxNt
    Ql_wdot = jnp.tensordot(
        D3hat,  # NpxNmxNb
        gust_dot,  # NbxNt
        axes=(2, 0),
    )  # NpxNmxNt
    return Q_w, Q_wdot, Q_wddot, Q_wsum, Ql_wdot


@Registry.register("GustRogerMc")
class GustRogerMc(Gust):
    def __init__(
        self, settings: intrinsicmodal.Dsystem, sol: solution.IntrinsicSolution
    ):
        self.settings = settings
        self.solaero = getattr(sol.data, f"modalaeroroger_{settings.name}")
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
        # simulation_time = self.settings.t
        self.collocation_points = self.settings.aero.gust.collocation_points
        self.npanels = len(self.collocation_points)
        self.dihedral = self.settings.aero.gust.panels_dihedral
        self.gust_length = self.settings.aero.gust.length
        self.gust_intensity = self.settings.aero.gust.intensity
        self.gust_totaltime = self.settings.aero.gust.totaltime
        self.xgust = self.settings.aero.gust.x
        self.time = self.settings.aero.gust.time
        self.ntime = self.settings.aero.gust.ntime
        #
        # (self.gust_totaltime, self.xgust, self.time, self.ntime) = intrinsicmodal.gust_discretisation(
        #     self.gust_intensity,
        #     self.dihedral,
        #     self.gust_shift,
        #     self.gust_step,
        #     self.settings.t,
        #     self.gust_length,
        #     self.u_inf,
        #     jnp.min(self.collocation_points[:, 0]),
        #     jnp.max(self.collocation_points[:, 0]),
        # )

        self.fshape_span = _get_spanshape(self.settings.aero.gust.shape)

    def calculate_downwash_unoptimised(self):
        """
        NpxNt panel downwash in time
        """
        # TODO: Optimize
        self.gust = jnp.zeros((self.npanels, self.ntime))
        self.gust_dot = jnp.zeros((self.npanels, self.ntime))
        self.gust_ddot = jnp.zeros((self.npanels, self.ntime))
        coeff = 2.0 * jnp.pi * self.u_inf / self.gust_length
        for panel in range(self.npanels):
            delay = (self.collocation_points[panel, 0] + self.gust_shift) / self.u_inf
            shape_span = self.shape_span(self.collocation_points[panel, 1])
            filter_time = jnp.where(
                (self.time >= delay) & (self.time <= delay + self.gust_totaltime), 1, 0
            )
            self.gust = self.gust.at[panel].set(
                filter_time
                * (
                    shape_span
                    * self.normals[panel]
                    * self.gust_intensity
                    / (self.u_inf * 2)
                    * (1 - jnp.cos(coeff * (self.time - delay)))
                )
            )
            self.gust_dot = self.gust_dot.at[panel].set(
                filter_time
                * (
                    shape_span
                    * self.normals[panel]
                    * self.gust_intensity
                    / (self.u_inf * 2)
                    * (jnp.sin(coeff * (self.time - delay)))
                    * coeff
                )
            )
            self.gust_ddot = self.gust_ddot.at[panel].set(
                filter_time
                * (
                    shape_span
                    * self.normals[panel]
                    * self.gust_intensity
                    / (self.u_inf * 2)
                    * (jnp.cos(coeff * (self.time - delay)))
                    * coeff**2
                )
            )
        self._define_eta()

    def calculate_downwash(self):
        """
        NpxNt panel downwash in time
        """
        self.gust, self.gust_dot, self.gust_ddot = _downwashRogerMc(
            self.u_inf,
            self.gust_length,
            self.gust_intensity,
            self.gust_shift,
            self.collocation_points,
            self.normals,
            self.time,
            self.gust_totaltime,
            self.fshape_span,
        )

        self._define_eta()

    def calculate_normals(self):
        if self.dihedral is not None:
            self.normals = self.dihedral

    def set_solution(self, sol: solution.IntrinsicSolution, sys_name: str):
        sol.add_container(
            "GustRoger",
            label="_" + sys_name,
            w=self.gust,
            wdot=self.gust_dot,
            wddot=self.gust_ddot,
            x=self.time,
            Qhj_w=self.Q_w,
            Qhj_wdot=self.Q_wdot,
            Qhj_wddot=self.Q_wddot,
            Qhj_wsum=self.Q_wsum,
            Qhjl_wdot=self.Ql_wdot,
        )

    def _define_eta(self):
        """
        NtxNm
        """

        (self.Q_w, self.Q_wdot, self.Q_wddot, self.Q_wsum, self.Ql_wdot) = _getGAFs(
            self.solaero.D0hat,  # NbxNm
            self.solaero.D1hat,
            self.solaero.D2hat,
            self.solaero.D3hat,
            self.gust,
            self.gust_dot,
            self.gust_ddot,
        )
