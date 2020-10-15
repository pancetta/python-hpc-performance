import itertools
from collections import defaultdict
import numpy as np

import benchmarks
from tools.registry import registry

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


def add_to_results(results, benchmark, rounds, durations):
    result = dict()
    result['name'] = benchmark.name
    result['rounds'] = rounds
    params = benchmark.params.__dict__.copy()
    params.pop('_FrozenClass__isfrozen')
    result['params'] = params
    # result['durations'] = durations[0:10]
    result['statistics'] = {'min': np.amin(durations),
                            'max': np.amax(durations),
                            'mean': np.mean(durations),
                            'median': np.median(durations),
                            'std': np.std(durations),
                            'var': np.var(durations)}

    id = benchmark.name
    for k, v in params.items():
        id += '_' + str(k) + '=' + str(v)
    c = 0
    id += '_' + str(c)
    while id in results:
        c += 1
        id = id.rpartition('_')[0] + '_' + str(c)
    results[id] = result

    return results


def gather_benchmarks():

    filter = {'partype': 'mpi', 'min_procs': 1}

    benchmarks = []
    for entry in registry:
        if filter.items() <= entry[1].items():
            keys, values = zip(*entry[2].items())
            list_of_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
            for params in list_of_params:
                benchmarks.append((entry[0], params, entry[1]))
    return benchmarks


def eval_results(results):

    grouped_results = defaultdict(list)
    for k, v in results.items():
        grouped_results[v['name']].append(k)

    for group, _ in grouped_results.items():
        print('-' * 10 + ' ' + group + ' ' + '-' * 10)
        sortkeys = sorted(grouped_results[group], key=lambda x: results[x]['statistics']['mean'])
        for key in sortkeys:
            ratio = results[key]['statistics']['mean'] / results[sortkeys[0]]['statistics']['mean']
            print(f"{key}:\t {results[key]['rounds']:6d}\t {results[key]['statistics']['mean']:10.6e} ({ratio:4.2f})")


def main():

    maxtime_per_benchmark = 1E-01
    maxrounds_per_benchmark = 10000

    benchmarks = gather_benchmarks()

    results = dict()
    for benchmark, bench_params, _ in benchmarks:
        b = benchmark(comm=comm, params=bench_params)
        durations = []
        rounds = 0
        sum_durations = 0
        while sum_durations < maxtime_per_benchmark and rounds < maxrounds_per_benchmark:
            rounds += 1
            b.reset()
            durations.append(b.run())
            if comm is not None:
                sum_durations += comm.allreduce(sendobj=durations[-1], op=MPI.SUM) / comm.Get_size()
            else:
                sum_durations += durations[-1]
            # print(sum(durations))
        # print(rounds, sum(durations), sum(durations) / rounds)
        results = add_to_results(results, b, rounds, durations)
        b.tear_down()

    eval_results(results)


if __name__ == '__main__':
    main()
