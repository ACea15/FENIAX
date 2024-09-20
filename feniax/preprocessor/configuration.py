import feniax.preprocessor.containers as containers
from feniax.preprocessor.containers.data_container import DataContainer
import feniax.preprocessor.utils as utils
import importlib
import feniax.preprocessor.inputs as inputs
import pathlib
from ruamel.yaml import YAML
import jax.numpy as jnp
import numpy as np
import argparse
import copy


class Config:
    def __init__(self, sett: dict):
        self.__sett = copy.copy(sett)
        self.__serial_data = None
        self.__extract_attr()
        self.__load_container()
        self.__build()
        self._data_dict = serialize(self)
        self.__defaults()
        self.__set_defaults()

    def __extract_attr(self):
        """Extracts attributes that do not belong to a container.
        This attributes are located at the first level of the input settings."""
        if "ex" in self.__sett.keys():
            self.__set_experimental(self.__sett.pop("ex"))
        if "engine" in self.__sett.keys():
            self.__set_attr(engine=self.__sett.pop("engine"))

    def __load_container(self):
        """Load the container with the configuration dataclasses"""

        # TODO: Extend to functionality for various containers
        self.__container = importlib.import_module(
            f"feniax.preprocessor.containers.{self.engine}"
        )
        self.__container = importlib.reload(self.__container)  # remove after testing

    def __defaults(self):
        self.__MOD_DEFAULT = dict(optionsjax=["jax_np", "jax_scipy"])
        self.__CONTAINER_DEFAULT = dict(intrinsicmodal="const")

    def __set_defaults(self):
        # default modules
        for k, v in self.__MOD_DEFAULT.items():
            _container = importlib.import_module(f"feniax.preprocessor.containers.{k}")
            for i in v:
                if not hasattr(self, i):
                    container_k = getattr(_container, "".join(["D", i]))
                    setattr(self, i, container_k())
        # default containers within self.engine module
        for k, v in self.__CONTAINER_DEFAULT.items():
            if self.engine == k:
                if not hasattr(self, v):
                    container_v = getattr(self.__container, "".join(["D", v]))
                    setattr(self, v, container_v())

    def __build(self):
        if self.engine == "intrinsicmodal":  # needs some specialisation here
            k = "fem"  # initialised fem first as it will be pass to system
            v = self.__sett[k]
            container_k = getattr(self.__container, "".join(["D", k]))
            container_k_initialised = container_k(**v)
            setattr(self, k, container_k_initialised)
            for k, v in self.__sett.items():
                if k != "fem":
                    container_k = getattr(self.__container, "".join(["D", k]))
                    if k == "systems" or k == "system":  # pass Dfem
                        container_k_initialised = container_k(
                            **(v | dict(_fem=self.fem))
                        )
                    else:
                        container_k_initialised = container_k(**v)
                    setattr(self, k, container_k_initialised)
        else:  # not currently in used
            for k, v in self.__sett.items():
                container_k = getattr(self.__container, "".join(["D", k]))
                setattr(self, k, container_k(**v))

    def __set_experimental(self, experimental: dict):
        ex_object = inputs.dict2object(experimental)
        setattr(self, "ex", ex_object)

    def __set_attr(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_file(cls, file_dir: str | pathlib.Path, **kwargs):
        yaml = YAML()
        yaml_dict = yaml.load(pathlib.Path(file_dir))
        return cls(yaml_dict)


class ValidateConfig:
    @staticmethod
    def validate(config):
        validate_engine = getattr(ValidateConfig, f"_{config.engine}")
        validate_engine(config)

    @staticmethod
    def _intrinsicmodal(config):
        assert hasattr(config, "driver"), "No 'driver' attr in config object"
        assert hasattr(config, "fem"), "No 'fem' attr in config object"

def serialize(obj: Config | DataContainer):
    dictionary = dict()
    for k, v in obj.__dict__.items():
        # serialise if it is ndarray
        if isinstance(v, jnp.ndarray) or isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, pathlib.Path):
            v = str(v)
        if k == "systems":
            dictionary[k] = dict(sett={})
            for k2, v2 in obj.systems.mapper.items():
                dictionary[k]["sett"][k2] = serialize(v2)
            continue
        # ensure the field is public
        if k[0] != "_":
            if isinstance(v, DataContainer):
                dictionary[k] = serialize(v)
            else:
                # ensure v is not an uninitialised field, which should not be saved
                if isinstance(obj, DataContainer):
                    if (
                        obj.__dataclass_fields__[k].init
                        and obj.__dataclass_fields__[k].metadata["yaml_save"]
                    ):
                        dictionary[k] = [
                            v,
                            # obj.__dataclass_fields__[k].metadata["description"],
                            obj.attributes.get(k, "No description available")
                        ]
                else:
                    dictionary[k] = [v, " "]
    return dictionary


def dump_to_yaml(file_out: str | pathlib.Path, config: Config, with_comments=True):
    yaml = YAML()
    file_out = pathlib.Path(file_out)
    file_out.parent.mkdir(parents=True, exist_ok=True)
    data = utils.dump_inputs(config._data_dict, with_comments=with_comments)
    with open(file_out, "w") as f:
        yaml.dump(data, f)


def initialise_config(
    input_file: str = None, input_dict: dict = None, input_obj: Config = None
) -> Config:
    if input_dict is None and input_obj is None:  # inputs given as .yaml file
        parser = argparse.ArgumentParser(
            prog="FENIAX",
            description="""This is the executable of Fininte-Element Models for
        Intrinsic Nonlinear Aeroelastic Simulations.""",
        )
        parser.add_argument(
            "input_file", help="path to the *.yaml input file", type=str, default=""
        )
        if input_file is not None:  # running from within python file
            config = Config.from_file(input_file)
            # parser.parse_args(input_file)
        else:  # running from command line
            args = parser.parse_args()
            config = Config.from_file(args.input_file)
    elif input_dict is not None and (
        input_file is None and input_obj is None
    ):  # inputs given as dict
        config = Config(input_dict)

    elif input_obj is not None and (
        input_file is None and input_dict is None
    ):  #  inputs directly as Config
        config = input_obj
    else:
        raise ValueError("Input error combination")
    ValidateConfig.validate(config)
    return config


if __name__ == "__main__":
    pass
