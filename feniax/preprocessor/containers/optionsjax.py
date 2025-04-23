from dataclasses import dataclass
from feniax.preprocessor.containers.data_container import DataContainer

import jax.lax

from feniax.preprocessor.utils import dfield, initialise_Dclass

def Ddataclass(cls):
    return dataclass(cls, frozen=True, kw_only=True)


@Ddataclass
class Djax_np(DataContainer):
    precision: jax.lax.Precision = dfield(
        "Precision in jnp operations", default=jax.lax.Precision.HIGHEST,
        yaml_save=False
    )
    allclose: dict[str:float] = dfield(
        """Relative and absolute tolerances""", default=dict(rtol=1e-4, atol=1e-4),
        yaml_save=False
    )

@Ddataclass
class Djax_scipy(DataContainer):
    eigh: dict = dfield("Eigen value solution", default=None, yaml_save=False)
