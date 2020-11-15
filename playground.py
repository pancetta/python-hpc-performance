import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

result_files = glob.glob('data/' + 'results*.json')


df = pd.read_json('data/results_macbookpro.json')

idx = df.groupby(['id'])['mean_duration'].transform(max) == df['mean_duration']
df_reduced = df[idx]
print(df_reduced)

# exit()


# df = pd.read_json('data/results_juwels.json')

# df = df[df.name.isin(['mpi_broadcast']) & df.params_n.isin([10000])]

for file in result_files:
    # df = pd.read_json(file)
    df = pd.read_json('data/results_macbookpro.json')
    idx = df.groupby(['id'])['mean_duration'].transform(max) == df['mean_duration']
    df = df[idx]

    scores = []
    for row, col in df.iterrows():
        data = np.array(col['stripped_timeline'])
        if data.size > 0:
            penalty = 1 - min(data.std() / data.mean(), 1)
        else:
            penalty = 1
        if penalty == 0:
            print('boom')
        # scores.append((col['sum_durations'] / col['mean_duration']) / col['sum_durations'] * penalty)
        # scores.append(1.0 / col['mean_duration'] * penalty)
        scores.append(col['MPI_size'] / col['mean_duration'] * penalty)
    print(file, np.median(scores))

exit()


