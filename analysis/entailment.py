import pandas as pd
from __init__ import *
from preprocessing.constants import MORALITY_ORIGIN

interviews = pd.read_pickle('data/cache/morality_origin.pkl')
interviews[MORALITY_ORIGIN].max(axis=0)

for origin in MORALITY_ORIGIN:
    print(origin)
    print(interviews[interviews[origin] == interviews[origin].max()]['Morality Origin'].iloc[0])
    print()
