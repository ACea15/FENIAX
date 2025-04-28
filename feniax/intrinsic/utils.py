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


class StateTrack:
    def __init__(self):
        self.states = dict()
        self.num_states = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.states[k] = jnp.arange(self.num_states, self.num_states + v)
            self.num_states += v

            
def build_systemstates(solution: str, target: str, bc1:str, rb_treatment: int, q0treatment: int, num_poles:int, num_modes: int, num_nodes: int):
    
    tracker = StateTrack()
    # TODO: keep upgrading/ add residualise
    if solution == "static" or solution == "staticAD":
        tracker.update(q2=num_modes)
        if target.lower() == "trim":
            tracker.update(qx=1)
    elif solution == "dynamic" or solution == "dynamicAD":
        tracker.update(q1=num_modes, q2=num_modes)
        if num_poles > 0:
            tracker.update(ql=num_poles * num_modes)
        if q0treatment == 1:
            tracker.update(q0=num_modes)
        if bc1.lower() != "clamped":
            if rb_treatment == 1:
                tracker.update(qr=4)
            elif rb_treatment == 2:
                tracker.update(qr=4 * num_nodes)
    elif solution == "multibody":
        tracker.update(q1=num_modes, q2=num_modes)
        if num_poles > 0:
            tracker.update(ql=num_poles * num_modes)
        if q0treatment == 1:
            tracker.update(q0=num_modes)
        if rb_treatment == 1:
            tracker.update(qr=4)
        elif rb_treatment == 2:
            tracker.update(qr=4 * num_nodes)
        
    return tracker

def build_gsystemstate(solution: str, target: str, bc1:str, rb_treatment: int, q0treatment: int, num_poles:int, num_modes: int, num_nodes: int):
    
    tracker = StateTrack()
    # TODO: keep upgrading/ add residualise
    if solution == "static" or solution == "staticAD":
        tracker.update(q2=num_modes)
        if target.lower() == "trim":
            tracker.update(qx=1)
    elif solution == "dynamic" or solution == "dynamicAD":
        tracker.update(q1=num_modes, q2=num_modes)
        if num_poles > 0:
            tracker.update(ql=num_poles * num_modes)
        if q0treatment == 1:
            tracker.update(q0=num_modes)
        if bc1.lower() != "clamped":
            if rb_treatment == 1:
                tracker.update(qr=4)
            elif rb_treatment == 2:
                tracker.update(qr=4 * num_nodes)
    elif solution == "multibody":
        tracker.update(q1=num_modes, q2=num_modes)
        if num_poles > 0:
            tracker.update(ql=num_poles * num_modes)
        if q0treatment == 1:
            tracker.update(q0=num_modes)
        if rb_treatment == 1:
            tracker.update(qr=4)
        elif rb_treatment == 2:
            tracker.update(qr=4 * num_nodes)
        
    return tracker
