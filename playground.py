import importlib

imports = ['numpy', 'scipy', 'numba', 'mpi4py']

modules = {}
for x in imports:
    try:
        modules[x] = importlib.import_module(x)
        print("Successfully imported ", x, '.')
    except ImportError:
        print("Error importing ", x, '.')

print([k + '==' + v.__version__ for k, v in modules.items()])