class Registry:
    def __init__(self, name, add_name_as_attr=False):
        self._name = name
        self._registry_dict = dict()
        self._add_name_as_attr = add_name_as_attr

    def _register(self, obj, name):
        if name in self._registry_dict:
            raise KeyError("{} is already registered in {}".format(name, self._name))
        self._registry_dict[name] = obj

    def register(self, name=None):
        def wrap(obj):
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            if self._add_name_as_attr:
                setattr(obj, "_registered_name", name)
            self._register(obj, cls_name)
            return obj

        return wrap

    def get(self, name):
        if name not in self._registry_dict:
            self._key_not_found(name)
        return self._registry_dict[name]

    def _key_not_found(self, name):
        raise KeyError("{} is unknown type of {} ".format(name, self._name))

    @property
    def registry_dict(self):
        return self._registry_dict
