from dataclasses import dataclass
import jax.lax

from fem4inas.preprocessor.utils import dfield, initialise_Dclass


@dataclass
class Djax_np:

    precision: jax.lax.Precision = dfield("Precision in jnp operations",
                                          default= jax.lax.Precision.HIGHEST)
    allclose: dict[str: float] = dfield("""Relative and absolute tolerances""",
                                        default=dict(rtol=1e-5,
                                                     atol=1e-6))
class Djax_scipy:

    eigh: dict = dfield("Eigen value solution",
                        default= jax.lax.Precision.HIGH)
