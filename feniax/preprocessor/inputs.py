from dataclasses import dataclass, field


class Inputs(dict):
    """Represents configuration options, works like a dict."""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            self[name] = Inputs({})
            return self[name]

    def __setattr__(self, name, val):
        self[name] = val


def dict2object(config: dict | Inputs):
    """Convert dictionary into instance allowing access dot notation."""
    if isinstance(config, dict):
        result = Inputs()
        for key in config:
            result[key] = dict2object(config[key])
        return result
    else:
        return config


if __name__ == "__main__":
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
