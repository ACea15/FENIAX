import fem4inas.preprocessor.containers as containers
from fem4inas.preprocessor.containers.data_container import DataContainer
from fem4inas.preprocessor.utils import dump_inputs
import importlib
import fem4inas.preprocessor.config as config
import pathlib
from ruamel.yaml import YAML
import jax.numpy as jnp
import numpy as np

class Config:

    def __init__(self, sett: dict):

        self.__sett = sett
        self.__serial_data = None
        self.__load_containers()        
        self.__extract_attr()
        self.__build()
        self._data_dict = serialize(self)

    def __extract_attr(self):
        """Extracts attributes that do not belong to a container"""
        if "ex" in self.__sett.keys():
            self.__set_experimental(self.__sett.pop('ex'))
        if "engine" in self.__sett.keys():
            self.__set_attr(engine=self.__sett.pop('engine'))

    def __load_containers(self):
        """Loads the containers"""

        # TODO: Extend to functionality for various containers
        self.__container = importlib.import_module(
            f"fem4inas.preprocessor.containers.{self.__sett['engine']}")
        self.__container = importlib.reload(self.__container) # remove after testing
        
    def __build(self):

        for k, v in self.__sett.items():
            container_k = getattr(self.__container, "".join(["D", k]))
            setattr(self, k, container_k(**v))

    def __set_experimental(self, experimental: dict):

        ex_object = config.dict2object(experimental)
        setattr(self, "ex", ex_object)

    def __set_attr(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        
    @classmethod
    def from_file(cls, file_dir: str|pathlib. Path, **kwargs):
        yaml = YAML()
        yaml_dict = yaml.load(file_dir)
        return cls(yaml_dict)

    # @staticmethod
    # def _serialize(obj):

    #     for k, v in obj.__dict__:
    #         if k[0] != "_":

def serialize(obj: Config | DataContainer):

    dictionary = dict()
    for k, v in obj.__dict__.items():
        # serialise if it is ndarray
        if isinstance(v, jnp.ndarray) or isinstance(v, np.ndarray):
            v = v.tolist()
        # ensure the field is public
        if k[0] != "_":
            if isinstance(v, DataContainer):
                dictionary[k] = serialize(v)
            else:
                # ensure v is not an uninitialised field, which should not be saved
                if (isinstance(obj, DataContainer) and
                    obj.__dataclass_fields__[k].init):
                    dictionary[k] = [v, obj.__dataclass_fields__[k].metadata['description']]
                else:
                    dictionary[k] = [v, " "]
    return dictionary

def dump_to_yaml(file_out, config: Config, with_comments=True):

    yaml = YAML()
    data = dump_inputs(config._data_dict, with_comments=with_comments)
    with open(file_out, "w") as f:
        yaml.dump(data, f)

if __name__ == "__main__":

    pass
