import pandas as pd
from sklearn.metrics import log_loss
from __init__ import *
from preprocessing.constants import MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings


def cross_entropy_loss(interviews):

    #Keep interviews with codings and merge 
    interviews = interviews[interviews['Wave'] == 1]
    interviews = merge_codings(interviews)
    interviews[[mo + '_y' for mo in MORALITY_ORIGIN]] = interviews[[mo + '_y' for mo in MORALITY_ORIGIN]].applymap(int)
    
    #Compute loss
    loss = pd.Series({mo: log_loss(interviews[mo + '_y'], interviews[mo + '_x']) for mo in MORALITY_ORIGIN})
    
    print(loss)
    print('Mean Loss:', loss.mean())


if __name__ == '__main__':
    config = [1]

    for c in config:
        if c == 1:
            interviews = pd.read_pickle('data/cache/morality_origin.pkl')
            cross_entropy_loss(interviews)