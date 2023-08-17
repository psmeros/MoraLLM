import pandas as pd
from __init__ import *
from sklearn.metrics import log_loss

from preprocessing.constants import MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings


def cross_entropy_loss(interviews):
    #Keep interviews with codings and merge 
    interviews = interviews[interviews['Wave'] == 1]
    interviews = merge_codings(interviews)
    interviews[[mo + '_y' for mo in MORALITY_ORIGIN]] = interviews[[mo + '_y' for mo in MORALITY_ORIGIN]].applymap(int)
    
    #Compute loss
    loss = pd.Series({mo: log_loss(interviews[mo + '_y'], interviews[mo + '_x']) for mo in MORALITY_ORIGIN})
    return loss


if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    models = ['md', 'lg', 'entail']

    for c in config:
        if c == 1:
            for model in models:
                interviews = pd.read_pickle('data/cache/morality_embeddings_'+model+'.pkl')
                loss = cross_entropy_loss(interviews)
                print(model, 'model mean loss:', loss.mean())