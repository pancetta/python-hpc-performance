from tools.frozen import FrozenClass


# short helper class to add params as attributes
class Params(FrozenClass):
    def __init__(self, params):
        # params.pop('__class__')
        # params.pop('self')
        for k, v in params.items():
            setattr(self, k, v)
        # freeze class, no further attributes allowed from this point
        self._freeze()
