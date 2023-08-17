import pandas as pd
from sklearn.metrics import log_loss
import spacy
import torch
from __init__ import *
from preprocessing.constants import MORALITY_ORIGIN
from preprocessing.embeddings import transform_embeddings
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
    #Hyperparameters
    config = [1]
    model = 'lg'
    embeddings_file = 'data/cache/morality_embeddings_'+model+'.pkl'
    transformation_matrix_file = 'data/cache/transformation_matrix_'+model+'.pkl'

    for c in config:
        if c == 1:
            interviews = pd.read_pickle('data/cache/morality_origin.pkl')
            cross_entropy_loss(interviews)
        elif c == 2:
            nlp = spacy.load('en_core_web_lg')

            #Load and transform data
            interviews = pd.read_pickle(embeddings_file)
            interviews = interviews[['Wave', 'Interview Code', 'R:Morality_Embeddings']].rename(columns={'R:Morality_Embeddings': 'Embeddings'})
            interviews['Embeddings'] = transform_embeddings(interviews['Embeddings'], transformation_matrix_file)

            #Compute cosine similarity with morality origin vectors
            morality_origin = pd.Series({mo:nlp(mo).vector for mo in MORALITY_ORIGIN})
            for mo in MORALITY_ORIGIN:
                interviews[mo] = interviews['Embeddings'].apply(lambda e: torch.cosine_similarity(torch.from_numpy(e).view(1, -1), torch.from_numpy(morality_origin[mo]).view(1, -1)).numpy()[0])
            interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN].apply(lambda x: pd.Series({mo:p for mo, p in zip(MORALITY_ORIGIN, torch.nn.functional.softmax(torch.from_numpy(x.to_numpy()), dim=0).numpy())}), axis=1)

            #Compute loss
            cross_entropy_loss(interviews)