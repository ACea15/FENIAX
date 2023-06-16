from dataclasses import dataclass, field


class Config(dict):
    """Represents configuration options, works like a dict."""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            self[name] = Config({})
            return self[name]

    def __setattr__(self, name, val):
        self[name] = val


def dict2object(config: dict | Config):
    """Convert dictionary into instance allowing access dot notation."""
    if isinstance(config, dict):
        result = Config()
        for key in config:
            result[key] = dict2object(config[key])
        return result
    else:
        return config


if __name__ == '__main__':

    d1 = {
        "conf1": {
            "key1": "aaa",
            "key2": 12321,
            "key3": {"a": 8},
        },
        "conf2": "bbbb",
    }

    c1 = dict2object(d1)

    def field1(description, options=None, default=None):

        return field(metadata={"description": description})

    @dataclass
    class try_meta2:
        v1: int

    @dataclass()
    class try_meta:
        v1: int = field(metadata={"description": "fff", "options": [1, 2, 3]})
        v2: int = field(metadata={"options": [1, 2]})
        v3: str
        v4: try_meta2 = field()

        def __post_init__(self):
            object.__setattr__(self, "v4", try_meta2(self.v4))

    # __dataclass_fields__['name'].metadata
    # uex = UsesExternalDict(9)
    # uex.__dataclass_fields__

    t1 = try_meta(1, 2, "hello", 5)

    sett = dict2object({})

    sett.engine = "IntrinsicModal"
    sett.driver.subcases = None
    sett.driver.supercases = None

    sett.fem.Ka_file = None
    sett.fem.Ma_file = None
    sett.fem.Ka = None
    sett.fem.Ma = None

    sett.simulation.type = "single"

    sett.geometry.file_name = None
    sett.geometry.input_data = None

    sett.quadratic_tensorterms = None

    sett.xloads.gravity = None
    sett.xloads.gravity_vect = None
    sett.xloads.follower_forces = None
    sett.xloads.dead_forces = None
    sett.xloads.gravity_forces = None
    sett.xloads.follower_forces = None
    sett.xloads.aero_forces = None

    sett.xloads.follower_points = None
    sett.xloads.dead_points = None
    sett.xloads.follower_interpolation = None
    sett.xloads.dead_interpolation = None

    sett.aero.u_inf = None
    sett.aero.rho_inf = None
    sett.aero.chord = None
    sett.aero.gafs = None
    sett.aero.poles = None
