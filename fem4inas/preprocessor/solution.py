from abc import ABC, abstractmethod
import pathlib
import jax.numpy as jnp
import numpy as np
import pickle


class Solution(ABC):
    @abstractmethod
    def set_solcontainer():
        ...

    def __init__(self, path: str | pathlib.Path = None):
        self.set_solcontainer()
        if path is not None:
            self.path = pathlib.Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
        self.containers = list()
        Data = type("Data", (), {})  # class container for dataclasses
        # located in self.solcontainer
        self.data = Data()

    def add_container(self, name: str, *args, label="", **kwargs):
        try:
            Container = getattr(self.solcontainer, name.capitalize())
        except AttributeError:
            raise AttributeError(
                f"Container {name} is not a valid name \
            in {self.solcontainer.__file__}"
            )
        setattr(self.data, name.lower() + label, Container(*args, **kwargs))
        self.containers.append(name + label)

    def load_container(self, name: str, label=""):
        try:
            Container = getattr(self.solcontainer, name.capitalize())
        except AttributeError:
            raise AttributeError(
                f"Container {name} is not a valid name \
            in {self.solcontainer.__file__}"
            )
        pathc = self.path / (name + label)
        solcontainer = load_container(pathc, Container)
        setattr(self.data, name.lower() + label, solcontainer)
        self.containers.append(name + label)

    def del_container(self, name, label=""):
        assert (
            name + label
        ) in self.containers, f"{name} is not a container in \
        the current solution object"
        delattr(self.data, name.lower() + label)
        self.containers.remove(name + label)

    def save_container(self, name: str, label="", del_obj: bool = False):
        assert (
            name + label
        ) in self.containers, f"{name} is not a container in \
        the current solution object"
        pathc = self.path / (name + label)
        pathc.mkdir(parents=True, exist_ok=True)
        container = getattr(self.data, name.lower() + label)
        save_container(pathc, container)
        if del_obj:
            self.del_container(name, label)

    def add_dict(self, name, label, obj):
        if not hasattr(self.data, name):
            setattr(self.data, name, dict())
        dattr = getattr(self.data, name)
        dattr[label] = obj

class IntrinsicSolution(Solution):
    def set_solcontainer(self):
        
        import fem4inas.preprocessor.containers
        self.sol_container = fem4inas.preprocessor.containers.intrinsicsol


def save_container(path, container):
    for attr_name in container.__slots__:
        attr = getattr(container, attr_name)
        attr_path = path / attr_name
        if isinstance(attr, jnp.ndarray):
            jnp.save(attr_path, attr)
        elif isinstance(attr, np.ndarray):
            jnp.save(attr_path, attr)
        elif isinstance(attr, (list, dict, tuple)):
            with open(attr_path, "wb") as fp:  # Pickling
                pickle.dump(attr, fp)
        else:
            raise ValueError(
                f"Not recognised attribute {attr_name} \
            type: {type(attr)}"
            )

def load_container(path: pathlib.Path, Container):
    container_path = path / Container.__name__.lower()
    kwargs = dict()
    for attr_name in Container.__slots__:
        attr_path = container_path / attr_name
        if Container.__annotations__[attr_name].__name__ == "Array":
            kwargs[attr_name] = jnp.load(attr_path)
        elif Container.__annotations__[attr_name].__name__ == "ndarray":
            kwargs[attr_name] = np.load(attr_path)
        elif (
            (Container.__annotations__[attr_name].__name__ == "dict")
            or (Container.__annotations__[attr_name].__name__ == "list")
            or (Container.__annotations__[attr_name].__name__ == "tuple")
        ):
            with open(attr_path, "rb") as fp:  # Unpickling
                kwargs[attr_name] = pickle.load(fp)
        else:
            raise ValueError(f"Not recognised type annotation {attr_name}")

    return Container(**kwargs)
