import itertools
from collections import defaultdict
import numpy as np
import time

import benchmarks
from tools.registry import registry

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


def add_to_results(results, benchmark, rounds, durations, overall_time):
    result = dict()
    result['name'] = benchmark.name
    result['rounds'] = rounds
    result['overall_time'] = overall_time
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

    # filter = {'partype': 'mpi', 'min_procs': 1}
    filter = {'partype': 'multithreaded', 'min_procs': 1}

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
            print(f"{key}:\t {results[key]['rounds']:6d}\t {results[key]['statistics']['mean']:10.6e} ({ratio:4.2f})\t"
                  f"{results[key]['overall_time']:6.4e}")


def main():

    maxtime_per_benchmark = 1E-00
    maxrounds_per_benchmark = 10000
    results = dict()

    benchmarks = gather_benchmarks()

    for benchmark_class, bench_params, _ in benchmarks:
        t0 = time.time()
        benchmark = benchmark_class(comm=comm, params=bench_params)
        durations = []
        rounds = 0
        sum_durations = 0

        while sum_durations < maxtime_per_benchmark and rounds < maxrounds_per_benchmark:
            rounds += 1
            benchmark.reset()
            duration = benchmark.run()
            if comm is not None:
                sum_durations += comm.allreduce(sendobj=duration, op=MPI.SUM) / comm.Get_size()
            else:
                sum_durations += duration
            durations.append(duration)

        t1 = time.time()
        results = add_to_results(results, benchmark, rounds, durations, t1 - t0)
        benchmark.tear_down()

    eval_results(results)


if __name__ == '__main__':
    main()
