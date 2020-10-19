import matplotlib.pyplot as plt
import argparse
import json
import numpy as np


def extract_timelines(results, filter):

    filtered_results = [d for d in results if filter.items() <= d.items()]
    # timelines = [d['timeline'] for d in filtered_results]
    if 'MPI_rank' in filtered_results[0]:
        ranks = [d['MPI_rank'] for d in filtered_results]
        timelines = np.array([d['timeline'][:-1] for d in filtered_results])
    else:
        ranks = [0]
        timelines = np.array([d['timeline'][:-1] for d in filtered_results])

    print(timelines)
    plt.figure()
    # plt.imshow(timelines, interpolation=None, cmap='Reds_r', aspect=4)
    plt.pcolormesh(timelines, cmap='Reds_r')
    plt.yticks([r + 0.5 for r in ranks], ranks)
    plt.xticks(np.arange(0.5, len(timelines[0]) + 1.5, 10), np.arange(0, filtered_results[0]['sum_durations'], 10 * filtered_results[0]['resolution']))
    plt.colorbar()
    # plt.plot(times, timelines)
    plt.show()

    return filtered_results[0]['timeline']



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='file', type=str, default='results.json',
                        help='input result file (default: results.json)')
    args = parser.parse_args()
    with open(args.file, "r") as read_file:
        results = json.load(read_file)

    filter = {'rounds': 187}
    filter = {"params_n": 1000, "params_m": 1000, "params_dtype": "float64"}
    filter = {"params_n": 100000}

    timelines = extract_timelines(results, filter)

    # print(len(timelines))


    # plt.show()
    # print(results)



if __name__ == '__main__':
    main()

