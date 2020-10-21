import itertools
import numpy as np
import time
import configparser
import argparse
import json
import platform
import psutil

from tools.registry import registry

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


def get_system_info():
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
    return info


def add_to_results(results, benchmark, rounds, durations, overall_time, analysis_resolution, comm):
    result = dict()
    result['name'] = benchmark.name

    info = get_system_info()
    for k, v in info.items():
        result['info_' + k] = v

    result['rounds'] = rounds
    result['overall_time'] = overall_time

    params = benchmark.params.__dict__.copy()
    params.pop('_FrozenClass__isfrozen')
    for k, v in params.items():
        result['params_' + k] = v

    # result['durations'] = durations
    result['min_duration'] = np.amin(durations)
    result['max_duration'] = np.amax(durations)
    result['mean_duration'] = np.mean(durations)
    result['median_duration'] = np.median(durations)
    result['std_duration'] = np.std(durations)
    result['var_duration'] = np.var(durations)

    if comm is not None and comm.Get_size() > 1:
        result['sum_durations'] = comm.allreduce(sum(durations), MPI.MAX)
    else:
        result['sum_durations'] = sum(durations)
    result['resolution'] = analysis_resolution
    result['timeline'] = np.histogram(np.cumsum(durations),
        bins=np.arange(0, result['sum_durations'] + result['resolution'], result['resolution']))[0].tolist()

    if comm is not None and comm.Get_size() > 1:
        rank = comm.Get_rank()
        result['MPI_rank'] = rank
        result['MPI_size'] = comm.Get_size()
        result_gather = comm.gather(result, root=0)
        if rank == 0:
            for result in result_gather:
                results.append(result)
    else:
        results.append(result)

    return results


def gather_benchmarks(filter):
    import benchmarks  # have to keep this to fill the registry needed below (magic happens in __init__.py)

    benchmarks = []
    for entry in registry:
        if filter.items() <= entry[1].items():
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

    results = []

    benchmarks = gather_benchmarks(dict(config['filter']))

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
            if comm.Get_size() > 1:
                sum_durations += comm.allreduce(sendobj=duration, op=MPI.SUM) / comm.Get_size()
            else:
                sum_durations += duration
            durations.append(duration)

        t1 = time.time()
        results = add_to_results(results, benchmark, rounds, durations, t1 - t0, analysis_resolution, comm)

        benchmark.tear_down()
    save_results(results)
    eval_results(results)


if __name__ == '__main__':
    main()
