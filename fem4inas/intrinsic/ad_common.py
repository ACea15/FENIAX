import fem4inas.intrinsic.objectives as objectives
import optimistix as optx
from functools import partial
import jax.numpy as jnp
import jax
import fem4inas.intrinsic.modes as modes
import fem4inas.intrinsic.couplings as couplings
import fem4inas.systems.sollibs.diffrax as diffrax
import fem4inas.systems.intrinsicSys as isys
import fem4inas.intrinsic.postprocess as postprocess
import equinox


def _compute_modes(X,
                  Ka,
                  Ma,
                  eigenvals,
                  eigenvecs,
                  config):

    modal_analysis = modes.shapes(X.T,
                                  Ka,
                                  Ma,
                                  eigenvals,
                                  eigenvecs,
                                  config
                                  )

    return modes.scale(*modal_analysis)

newton = partial(jax.jit, static_argnames=['F', 'sett'])(diffrax.newton)

_solve = partial(jax.jit, static_argnames=['eqsolver', 'dq', 'sett'])(isys._staticSolve)
