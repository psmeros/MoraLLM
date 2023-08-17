import numpy as np
import pandas as pd
from __init__ import *
from sklearn.linear_model import LinearRegression
from transformers import pipeline

from preprocessing.constants import MORALITY_ORIGIN
from preprocessing.embeddings import transform_embeddings

#Zero-shot classification on the morality origin classes
def zero_shot_classification(interviews):
    #Premise and hypothesis templates
    hypothesis_template = 'The morality origin is {}.'
    morality_pipeline =  pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    result_dict = lambda l: pd.DataFrame([{l:s for l, s in zip(r['labels'], r['scores'])} for r in l])
    morality_origin = result_dict(morality_pipeline(interviews['Morality Origin'].tolist(), MORALITY_ORIGIN, hypothesis_template=hypothesis_template))

    #Join and filter results
    interviews = interviews.join(morality_origin)
    return interviews

#Compute coefficients for transforming waves
def regression(from_wave, to_wave, temporal_interviews, moral_foundations):
    samples = temporal_interviews[to_wave + ':R:Morality_Embeddings'] - temporal_interviews[from_wave + ':R:Morality_Embeddings']
    samples = samples.apply(lambda x: x/np.linalg.norm(x))

    X = moral_foundations['Embeddings'].apply(pd.Series).T.apply(list, axis=1).tolist()
    scores = []
    coefs = []
    for y in samples:
        regr = LinearRegression(fit_intercept=False)
        regr.fit(X, y)
        scores += [regr.score(X, y)]
        coefs += [regr.coef_]

    coefs = pd.DataFrame(coefs).set_axis(labels=moral_foundations['Name'], axis=1)
    coefs['Score'] = scores
    return coefs

if __name__ == '__main__':
    #Hyperparameters
    config = [2]

    for c in config:
        if c == 1:
            model = 'lg'

            moral_foundations_file = 'data/cache/moral_foundations_'+model+'.pkl'
            transformation_matrix_file = 'data/cache/transformation_matrix_'+model+'.pkl'
            temporal_embeddings_file = 'data/cache/temporal_morality_embeddings_'+model+'.pkl'
            from_wave = 'Wave 1'
            to_wave = 'Wave 3'

            #Load data
            moral_foundations = pd.read_pickle(moral_foundations_file)
            temporal_interviews = pd.read_pickle(temporal_embeddings_file)
            
            #Τransform embeddings
            moral_foundations['Embeddings'] = transform_embeddings(moral_foundations['Embeddings'], transformation_matrix_file)
            for wave in ['Wave 1', 'Wave 2', 'Wave 3']:
                temporal_interviews[wave+':R:Morality_Embeddings'] = transform_embeddings(temporal_interviews[wave+':R:Morality_Embeddings'], transformation_matrix_file)

            regression(from_wave, to_wave, temporal_interviews, moral_foundations)