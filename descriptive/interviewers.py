import re
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocessing.embeddings import compute_embeddings


# Find k interviewers with maximum distance
def k_most_distant_interviewers(interviewers_embeddings, k):

    # Calculate pairwise distances between embeddings
    distances = pdist(interviewers_embeddings.tolist(), metric='cosine')
    pairwise_distances = squareform(distances)

    # Iterate through all combinations and find the one with maximum sum of distances
    combinations_k = combinations(range(len(interviewers_embeddings)), k)

    max_sum_distances = -np.inf
    optimal_combination = None

    for combination in combinations_k:
        sum_distances = np.sum(pairwise_distances[np.ix_(combination, combination)])
        if sum_distances > max_sum_distances:
            max_sum_distances = sum_distances
            optimal_combination = combination

    return list(optimal_combination)


#Plot embeddings of k distant interviewers
def plot_embeddings(embeddings_file, section, wave=1, dim_reduction='TSNE', perplexity=5, k=3):

    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[interviews['Wave'] == wave]
    interviews = interviews[['Name of Interviewer', section]].dropna().reset_index(drop=True)

    interviewers = interviews.groupby('Name of Interviewer').mean()

    interviewers_indices = k_most_distant_interviewers(interviewers[section], k)
    selected_interviewers = interviewers.iloc[interviewers_indices].index.tolist()

    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))

    if dim_reduction == 'TSNE':
        data = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(interviews[section].apply(pd.Series))
    elif dim_reduction == 'PCA':
        data = PCA(n_components=2, random_state=42).fit_transform(interviews[section].apply(pd.Series))

    data = interviews[['Name of Interviewer']].join(pd.DataFrame(data))
    data = data[data['Name of Interviewer'].isin(selected_interviewers)]
    ax = sns.kdeplot(data=data, x=0, y=1, hue='Name of Interviewer', fill=True, alpha=0.5, bw_adjust=1)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Top ' + str(k) + ' Most Distant Interviewers on '+ re.split('[:_]', section)[1])
    plt.gca().get_legend().set_title('Interviewer')
    plt.savefig('data/plots/interviewer_embeddings.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # compute_embeddings('data/waves', 'data/cache/interviewer_embeddings.pkl', 'I:Morality', model='lg')
    plot_embeddings(embeddings_file='data/cache/interviewer_embeddings.pkl', section='I:Morality_Embeddings')