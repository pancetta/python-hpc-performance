import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

result_files = glob.glob('data/' + 'results*.json')

# df = pd.read_json('data/results_macbookpro.json')

for file in result_files:
    # df = pd.read_json(file)
    df_full = pd.read_json('data/results_macbookpro.json')

    df_seq = df_full[df_full['partype'] == 'sequential']
    penalties_seq = df_seq['timeline'].apply(np.asarray) \
        .apply(lambda x: x[x > 0][2:-2]) \
        .apply(lambda x: 1 - min(x.std() / x.mean(), 1) if len(x) > 0 else 1)
    scores_seq = df_seq['MPI_size'] / df_seq['mean_duration'] * penalties_seq

    df_mt = df_full[df_full['partype'] == 'multithreaded']
    penalties_mt = df_mt['timeline'].apply(np.asarray) \
        .apply(lambda x: x[x > 0][2:-2]) \
        .apply(lambda x: 1 - min(x.std() / x.mean(), 1) if len(x) > 0 else 1)
    scores_mt = df_mt['MPI_size'] / df_mt['mean_duration'] * penalties_mt
    
    df_par = df_full[df_full['partype'] == 'mpi']
    idx = df_par.groupby(['id'])['mean_duration'].transform(max) == df_par['mean_duration']
    df_par = df_par[idx]
    penalties_par = df_par['timeline'].apply(np.asarray)\
        .apply(lambda x: x[x > 0][2:-2])\
        .apply(lambda x: 1 - min(x.std() / x.mean(), 1) if len(x) > 0 else 1)
    # scores = 1.0 / df['mean_duration'] * penalties
    scores_par = df_par['MPI_size'] / df_par['mean_duration'] * penalties_par
    print(file, np.median(scores_seq), np.median(scores_mt), np.median(scores_par))

exit()


