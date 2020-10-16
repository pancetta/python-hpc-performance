import re

file = 'bench_run/000039/sub_step_shared/results_nranks=4_test=aaa.json'
patterns = re.search('results_(.*).json', file)[1].split('_')

d = dict()
for pattern in patterns:
    k, v = pattern.split('=')
    d[k] = v
print(d)
# print(patterns
