import argparse
import glob
import json
import re


def augment_and_join_result_dicts(result_files):
    results = []
    for file in result_files:
        with open(file, "r") as read_file:
            result = json.load(read_file)
        for item in result:
            patterns = re.search('results_(.*).json', file)
            if patterns is not None:
                patterns = patterns[1].split('_')
                for pattern in patterns:
                    k, v = pattern.split('=')
                    item['run_' + k] = v
            results.append(item)
    return results


def eval_results(results):

    for name in set(r['name'] for r in results):
        print('-' * 10 + ' ' + name + ' ' + '-' * 10)
        sortedgroup = sorted(filter(lambda item: item.get('name') == name, results), key=lambda x: x['mean_duration'])
        for res in sortedgroup:
            ratio = res['mean_duration'] / sortedgroup[0]['mean_duration']
            param_str = {**{key: res[key] for key in res if key.startswith('params')}, **{key: res[key] for key in res if key.startswith('run')}}
            print(f"{param_str}:\t "
                  f"{res['rounds']:6d}\t "
                  f"{res['mean_duration']:6.4e} ({ratio:4.2f})\t "
                  f"{res['overall_time']:6.4e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dir', type=str, default='.',
                        help='input directory for the result files (default: .)')
    args = parser.parse_args()
    result_files = glob.glob(args.dir + '/' + 'results*.json')

    results = augment_and_join_result_dicts(result_files)
    eval_results(results)


if __name__ == '__main__':
    main()