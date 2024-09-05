class Registry:
    _registry = {}

    @classmethod
    def register(cls, key):
        def decorator(factory_class):
            print(f"***** Registering {key} *****")
            cls._registry[key] = factory_class
            return factory_class

        return decorator

    @classmethod
    def create_instance(cls, key, *args, **kwargs):
        if key in cls._registry:
            print(f"***** Creating instance of {key} *****")
            factory_class = cls._registry[key]
            return factory_class(*args, **kwargs)
        else:
            raise KeyError(f"Class '{key}' not found in the registry")
