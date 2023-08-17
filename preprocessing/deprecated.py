from itertools import combinations
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from __init__ import *
from sklearn.linear_model import LinearRegression

from preprocessing.embeddings import transform_embeddings

#Compute coefficients for transforming waves
def regression(from_wave='Wave 1', to_wave='Wave 3', model='lg'):
    moral_foundations_file = 'data/cache/moral_foundations_'+model+'.pkl'
    transformation_matrix_file = 'data/cache/transformation_matrix_'+model+'.pkl'
    temporal_embeddings_file = 'data/cache/temporal_morality_embeddings_'+model+'.pkl'

    #Load data
    moral_foundations = pd.read_pickle(moral_foundations_file)
    temporal_interviews = pd.read_pickle(temporal_embeddings_file)
    
    #Τransform embeddings
    moral_foundations['Embeddings'] = transform_embeddings(moral_foundations['Embeddings'], transformation_matrix_file)
    for wave in ['Wave 1', 'Wave 2', 'Wave 3']:
        temporal_interviews[wave+':R:Morality_Embeddings'] = transform_embeddings(temporal_interviews[wave+':R:Morality_Embeddings'], transformation_matrix_file)

    #Compute coefficients
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

# Find k embeddings with maximum distance
def k_most_distant_embeddings(embeddings, k):

    # Calculate pairwise distances between embeddings
    distances = pdist(embeddings.tolist(), metric='cosine')
    pairwise_distances = squareform(distances)

    # Iterate through all combinations and find the one with maximum sum of distances
    combinations_k = combinations(range(len(embeddings)), k)

    max_sum_distances = -np.inf
    optimal_combination = None

    for combination in combinations_k:
        sum_distances = np.sum(pairwise_distances[np.ix_(combination, combination)])
        if sum_distances > max_sum_distances:
            max_sum_distances = sum_distances
            optimal_combination = combination

    return list(optimal_combination)