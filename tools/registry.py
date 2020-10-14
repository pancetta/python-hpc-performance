registry = []


def register(cls, bench_type=None, bench_params=None):
    registry.append((cls, bench_type, bench_params))
    return cls
