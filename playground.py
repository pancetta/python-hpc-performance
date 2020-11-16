import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

result_files = glob.glob('data/' + 'results*.json')

# df = pd.read_json('data/results_macbookpro.json')

for file in result_files:
    df_full = pd.read_json(file)

    df_seq = df_full[df_full['partype'] == 'sequential']
    # Why filter by id? Could be that we ran the sequential benchmarks also with multiple cores to test other things.
    # Here we take the fastest sequential run, whatever this may be.
    idx = df_seq.groupby(['id'])['mean_duration'].transform(min) == df_seq['mean_duration']
    df_seq = df_seq[idx]
    penalties_seq = df_seq['timeline'].apply(np.asarray) \
        .apply(lambda x: x[x > 0][2:-2]) \
        .apply(lambda x: 1 - min(x.std() / x.mean(), 1) if len(x) > 0 else 1)
    scores_seq = 1.0 / df_seq['mean_duration'] * penalties_seq

    df_mt = df_full[df_full['partype'] == 'multithreaded']
    idx = df_mt.groupby(['id'])['mean_duration'].transform(min) == df_mt['mean_duration']
    df_mt = df_mt[idx]
    penalties_mt = df_mt['timeline'].apply(np.asarray) \
        .apply(lambda x: x[x > 0][2:-2]) \
        .apply(lambda x: 1 - min(x.std() / x.mean(), 1) if len(x) > 0 else 1)
    scores_mt = 1.0 / df_mt['mean_duration'] * penalties_mt
    
    df_par = df_full[df_full['partype'] == 'mpi']
    idx = df_par.groupby(['id'])['mean_duration'].transform(max) == df_par['mean_duration']
    df_par = df_par[idx]
    penalties_par = df_par['timeline'].apply(np.asarray)\
        .apply(lambda x: x[x > 0][2:-2])\
        .apply(lambda x: 1 - min(x.std() / x.mean(), 1) if len(x) > 0 else 1)
    # Why multipy by MPI_Size? Doesn't matter for scaling tests, but rewards stress tests with more cores.
    scores_par = df_par['MPI_size'] / df_par['mean_duration'] * penalties_par

    print(file, np.median(scores_seq), np.median(scores_mt), np.median(scores_par))

exit()


