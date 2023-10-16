from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

@dataclass(slots=True)
class Modes:
    phi1: jnp.ndarray
    psi1: jnp.ndarray
    phi2: jnp.ndarray
    phi1l: jnp.ndarray
    phi1ml: jnp.ndarray
    psi1l: jnp.ndarray
    phi2l: jnp.ndarray
    psi2l: jnp.ndarray
    omega: jnp.ndarray    
    X_xdelta: jnp.ndarray
    C0ab: jnp.ndarray
    C06ab: jnp.ndarray

@dataclass(slots=True)
class Couplings:
    alpha1: jnp.ndarray
    alpha2: jnp.ndarray
    gamma1: jnp.ndarray
    gamma2: jnp.ndarray

@dataclass(slots=True)
class DynamicSystem:
    q: jnp.ndarray
    X1: jnp.ndarray
    X2: jnp.ndarray
    X3: jnp.ndarray
    Cab: jnp.ndarray
    ra: jnp.ndarray
    
@dataclass(slots=True)
class StaticSystem:
    q: jnp.ndarray
    X2: jnp.ndarray = None
    X3: jnp.ndarray = None
    Cab: jnp.ndarray = None
    ra: jnp.ndarray = None

@dataclass(slots=True)
class ModalAeroRoger:

    poles: jnp.ndarray = None
    A0: jnp.ndarray = None
    A1: jnp.ndarray = None
    A2: jnp.ndarray = None
    Ap: jnp.ndarray = None
    B0: jnp.ndarray = None
    B1: jnp.ndarray = None
    B2: jnp.ndarray = None
    Bp: jnp.ndarray = None
    D0: jnp.ndarray = None
    D1: jnp.ndarray = None
    D2: jnp.ndarray = None
    Dp: jnp.ndarray = None
    C0: jnp.ndarray = None

# import dataclasses    
# field_types = {field.name: field.type for field in dataclasses.fields(Modes)}
