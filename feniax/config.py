from dataclasses import dataclass


class ConfigObject(dict):
    """
    Represents configuration options' group, works like a dict
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, val):
        self[name] = val


def config2object(config):
    """
    Convert dictionary into instance allowing access to dictionary keys using
    dot notation (attributes).
    """
    if isinstance(config, dict):
        result = ConfigObject()
        for key in config:
            result[key] = config2object(config[key])
        return result
    else:
        return config


d1 = {
    "conf1": {
        "key1": "aaa",
        "key2": 12321,
        "key3": {"a": 8},
    },
    "conf2": "bbbb",
}

c1 = config2object(d1)


@dataclass
class Book:
    title: str
    price: int
    author: str = "Unknown author"


data = {
    "title": "Fahrenheit 451",
    "price": 100,
}

factory = dataclass_factory.Factory()
book: Book = factory.load(data, Book)  # Same as Book(title="Fahrenheit 451", price=100)
serialized = factory.dump(book)

from typing import Any


@dataclass
class Field:
    value: Any
