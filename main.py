import itertools
import time
import configparser
import argparse
import json
import platform
import psutil
import re
import importlib
import hashlib

import numpy as np

from mpi4py import MPI

from tools.registry import registry


def get_system_info():

    imports = ['numpy', 'scipy', 'numba', 'mpi4py']

    info = dict()
    info['platform'] = platform.system()
    info['platform-release'] = platform.release()
    info['platform-version'] = platform.version()
    info['architecture'] = platform.machine()
    info['processor'] = platform.processor()
    info['physical-cores'] = psutil.cpu_count(logical=False)
    info['total-cores'] = psutil.cpu_count(logical=True)
    info['cpu-frequency'] = (psutil.cpu_freq().min, psutil.cpu_freq().max)
    info['ram'] = str(round(psutil.virtual_memory().total / (1024.0 ** 3)))+" GB"
    info['python-version'] = platform.python_version()

    for x in imports:
        mod = importlib.import_module(x)
        info['module_' + x] = mod.__version__

    return info


def add_to_results(results, bench_params, bench_type, rounds, durations, overall_time, analysis_resolution, comment, comm):
    result = dict()

    result['name'] = bench_type['name']
    result['comment'] = comment
    result['partype'] = bench_type['partype']

    info = get_system_info()
    for k, v in info.items():
        result['info_' + k] = v

    result['rounds'] = rounds
    result['overall_time'] = overall_time

    params = bench_params.copy()
    for k, v in params.items():
        result['params_' + k] = v

    # result['durations'] = durations
    result['min_duration'] = np.amin(durations)
    result['max_duration'] = np.amax(durations)
    result['mean_duration'] = np.mean(durations)
    result['median_duration'] = np.median(durations)
    result['std_duration'] = np.std(durations)
    result['var_duration'] = np.var(durations)
    result['sum_durations'] = comm.allreduce(sum(durations), MPI.MAX)
    result['resolution'] = analysis_resolution

    timeline = np.histogram(np.cumsum(durations), bins=np.arange(0, result['sum_durations'] + result['resolution'],
                                                                 result['resolution']))[0]
    result['timeline'] = timeline.tolist()
    result['stripped_timeline'] = timeline[timeline > 0][2:-2].tolist()

    rank = comm.Get_rank()
    size = comm.Get_size()
    result['MPI_rank'] = rank
    result['MPI_size'] = size

    m = hashlib.md5()
    encoded = json.dumps({**bench_type, **bench_params, 'MPI_size': size}, sort_keys=True).encode()
    m.update(encoded)
    result['id'] = m.hexdigest()

    result_gather = comm.gather(result, root=0)
    if rank == 0:
        for result in result_gather:
            results.append(result)
    else:
        results = []

    return results


def gather_benchmarks(filter):
    import benchmarks  # have to keep this to fill the registry needed below (magic happens in __init__.py)

    benchmarks = []
    for entry in registry:
        if all([re.match(v, entry[1][k]) for k, v in filter.items()]):
            if bool(entry[2]):
                keys, values = zip(*entry[2].items())
                list_of_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
                for params in list_of_params:
                    benchmarks.append((entry[0], params, entry[1]))
            else:
                benchmarks.append((entry[0], {}, entry[1]))
    return benchmarks


def eval_results(results):

    for name in set(r['name'] for r in results):
        print('-' * 10 + ' ' + name + ' ' + '-' * 10)
        sortedgroup = sorted(filter(lambda item: item.get('name') == name, results), key=lambda x: x['mean_duration'])
        for res in sortedgroup:
            ratio = res['mean_duration'] / sortedgroup[0]['mean_duration']
            print(f"{ {key: res[key] for key in res if key.startswith('params')} }:\t "
                  f"{res['rounds']:6d}\t "
                  f"{res['mean_duration']:6.4e} ({ratio:4.2f})\t "
                  f"{res['overall_time']:6.4e}")


def save_results(results):
    fname = 'results.json'
    with open(fname, "w") as write_file:
        json.dump(results, write_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='filename', type=str, default='example.ini',
                        help='input file for the configuration parameters (default: example.ini)')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.filename)

    maxtime_per_benchmark = config['main'].getfloat('maxtime_per_benchmark')
    maxrounds_per_benchmark = config['main'].getint('maxrounds_per_benchmark')
    analysis_resolution = config['main'].getfloat('analysis_resolution')
    comment = config['main']['comment']

    results = []
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    benchmarks = gather_benchmarks(dict(config['filter']))
    k = 0
    nbench = len(benchmarks)
    overall_time = 0.0

    if rank == 0:
        print(f'Starting {nbench} benchmarks, will need at least {nbench * maxtime_per_benchmark:.2f} sec.', flush=True)
        print()

    tbegin = time.time()
    for benchmark_class, bench_params, bench_type in benchmarks:
        t0 = time.time()
        benchmark = benchmark_class(name=bench_type['name'], params=bench_params, comm=comm)
        durations = []
        rounds = 0
        sum_durations = 0

        while sum_durations < maxtime_per_benchmark and rounds < maxrounds_per_benchmark:
            rounds += 1
            benchmark.reset()
            duration = benchmark.run()
            sum_durations += comm.allreduce(sendobj=duration, op=MPI.SUM) / comm.Get_size()
            durations.append(duration)

        t1 = time.time()
        results = add_to_results(results, bench_params, bench_type, rounds, durations, t1 - t0, analysis_resolution, comment, comm)

        benchmark.tear_down()
        k += 1
        overall_time += t1 - t0

        if rank == 0:
            print(f'   done with {bench_type["name"]}, {k} / {nbench} benchmarks, est. time left: '
                  f'{(nbench / k  - 1) * overall_time:.2f} / {nbench / k * overall_time:.2f} sec.', flush=True)

    tend = time.time()

    if rank == 0:
        print()
        print(f'Finished {nbench} benchmarks after {tend - tbegin:.2f} sec., overhead: '
              f'{100 * ((tend - tbegin) / (nbench * maxtime_per_benchmark) - 1):.2f}%', flush=True)
        print()

        save_results(results)
        eval_results(results)


if __name__ == '__main__':
    main()
