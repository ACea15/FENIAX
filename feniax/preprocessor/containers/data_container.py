import re
from dataclasses import fields

class DataContainer:
    attributes = {}
    _attributes_initialized = False

    # def __new__(cls, *args, **kwargs):
    #     if not cls._attributes_initialized:
    #         cls._initialize_attributes()
    #     return super().__new__(cls)

    def set_value(self, name, value):
        self.__setattr__(self, name, value)
        
    @classmethod
    def _initialize_attributes(cls):
        """
        Parses the docstring of the class and places the description to the dictionary attributes
        """
        docstring = cls.__doc__
        attributes = {}
        in_attributes_section = False

        lines = docstring.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "Attributes" or line == "Parameters":
                in_attributes_section = True
                continue
            if in_attributes_section and line.startswith('----'):
                continue
            if in_attributes_section:
                if line == "":  # Exit section once we reach an empty line
                    break
                # Look ahead to pick the description
                if i + 1 < len(lines) and re.match(r".+:", lines[i]):
                    if re.match(r".+:", lines[i+1]):
                        break
                    name_type = line.split(':')
                    attribute_name = name_type[0].strip()
                    attribute_type = name_type[1].strip()
                    attribute_description = lines[i + 1].strip()
                    #attributes.append(f"{attribute_name} ({attribute_type}): {attribute_description}")
                    attributes[attribute_name] = attribute_description
        cls.attributes = attributes
        cls._attributes_initialized = True

    def serialize(self): ...
