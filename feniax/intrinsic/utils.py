import jax.numpy as jnp
import pathlib
import logging

class Registry:
    _registry = {}

    @classmethod
    def register(cls, key):
        def decorator(factory_class):
            #print(f"***** Registering {key} *****")
            cls._registry[key] = factory_class
            return factory_class

        return decorator

    @classmethod
    def create_instance(cls, key, *args, **kwargs):
        if key in cls._registry:
            logging.info(f"Creating instance of {key}")
            factory_class = cls._registry[key]
            return factory_class(*args, **kwargs)
        else:
            raise KeyError(f"Class '{key}' not found in the registry")

def compute_eigs_load(
    num_modes: int, path: pathlib.Path, eig_names: list[str], *args, **kwargs
) -> (jnp.ndarray, jnp.ndarray):
    if path is not None:
        eigenvals = jnp.load(path / eig_names[0])
        eigenvecs = jnp.load(path / eig_names[1])
    else:
        eigenvals = jnp.load(eig_names[0])
        eigenvecs = jnp.load(eig_names[1])
    reduced_eigenvals = eigenvals[:num_modes]
    reduced_eigenvecs = eigenvecs[:, :num_modes]
    return reduced_eigenvals, reduced_eigenvecs
        
