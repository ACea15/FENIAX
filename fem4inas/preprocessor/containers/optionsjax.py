from dataclasses import dataclass
import jax.lax

from fem4inas.preprocessor.utils import dfield, initialise_Dclass


@dataclass
class Djax_np:

    precision: jax.lax.Precision = dfield("Precision in tensor and dot products",
                                          default= jax.lax.Precision.HIGH)
class Djax_scipy:

    eigh: dict = dfield("Eigen value solution",
                        default= jax.lax.Precision.HIGH)
