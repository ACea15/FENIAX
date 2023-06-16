from dataclasses import field


def field1(description):

    return field(metadata={"description": description})


def field2(description, default):

    if isinstance(default, (str, int, bool, float, tuple)):
        return field(default=default, metadata={"description": description})
    else:
        return field(
            default_factory=(lambda: default), metadata={"description": description}
        )


def field2o(description, options):

    return field(metadata={"description": description, "options": options})


def field3(description, default, options):

    if isinstance(default, (str, int, bool, float, tuple)):
        return field(
            default=default, metadata={"description": description, "options": options}
        )
    else:
        return field(
            default_factory=(lambda: default),
            metadata={"description": description, "options": options},
        )
