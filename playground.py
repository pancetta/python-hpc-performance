# from benchmarks.benchmarks import Benchmarks
import benchmarks
from tools.registry import registry
import itertools




print(registry)

keys, values = zip(*registry[1][2].items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(permutations_dicts)

# # print([obj() for obj in Benchmarks.__subclasses__()])
# for k, v in registry.items():
#     print(v.name)
#
# print([b() for _, b in registry.items() if b.name == 'matmul'])
# # print(registry)