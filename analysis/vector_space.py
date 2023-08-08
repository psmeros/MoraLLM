from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocessing.constants import MORALITY_ORIGIN
from preprocessing.embeddings import transform_embeddings


#Plot morality embeddings of all waves
def plot_morality_embeddings(interviews, dim_reduction='TSNE', perplexity=5):
    #Reduce dimensionality
    dim_reduction_alg = TSNE(n_components=2, perplexity=perplexity, random_state=42) if dim_reduction == 'TSNE' else PCA(n_components=2, random_state=42) if dim_reduction == 'PCA' else None
    interviews = interviews[['Name']].join(pd.DataFrame(dim_reduction_alg.fit_transform(interviews['Embeddings'].apply(pd.Series))))
    interviews = interviews.dropna()

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))
    color_palette = sns.color_palette('Set2')
    sns.kdeplot(data=interviews, x=0, y=1, hue='Name', fill=True, alpha=0.5, hue_order=['Wave 1', 'Wave 2', 'Wave 3'], palette=color_palette[:3])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Morality Embeddings')
    plt.savefig('data/plots/morality-embeddings.png', bbox_inches='tight')
    plt.show()

#Plot moral foundations by wave
def plot_moral_foundations(interviews, moral_foundations):    
    #Compute similarity
    interviews = interviews.merge(moral_foundations, how='cross')
    interviews['Similarity'] = interviews.apply(lambda x: 1 - cosine(x['Embeddings_x'], x['Embeddings_y']), axis=1)
    interviews = interviews[['Name_x', 'Name_y', 'Similarity']].rename(columns={'Name_x': 'Wave', 'Name_y': 'Anchor'})

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 20))
    ax = sns.boxplot(data=interviews, y='Anchor', x='Similarity', hue='Wave', hue_order=['Wave 1', 'Wave 2', 'Wave 3'], orient='h', palette='Set2')
    ax.legend(title='', loc='upper center', bbox_to_anchor=(0.3, -0.03), ncol=3)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Moral Foundations by Wave')
    plt.savefig('data/plots/morality-moral_foundations.png', bbox_inches='tight')
    plt.show()

    #Aggregate similarity
    print(interviews.groupby('Wave').mean(numeric_only=True))

#Plot semantic shift by morality origin
def plot_semantic_shift(interviews, wave_list=['Wave 1', 'Wave 2', 'Wave 3']):
    #Compute semantic shift between all pairs of waves
    compute_shift = lambda i: np.sum([cosine(i[w[0] + ':R:Morality_Embeddings'], i[w[1] + ':R:Morality_Embeddings']) for w in list(combinations(wave_list, 2))])
    interviews['Shift'] = interviews.apply(compute_shift, axis=1)

    #Prepare data for plotting
    interviews = interviews[MORALITY_ORIGIN + ['Shift']]
    interviews = interviews.melt(id_vars=['Shift'], value_vars=MORALITY_ORIGIN, var_name='Morality Origin', value_name='Check')
    interviews = interviews[interviews['Check'] == True].drop(columns=['Check'])

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 10))
    ax = sns.boxplot(data=interviews, y='Morality Origin', x='Shift', orient='h', palette='Set2')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Semantic Shift by Morality Origin')
    plt.savefig('data/plots/morality-semantic_shift.png', bbox_inches='tight')
    plt.show()

    #Print order by median
    print(' < '.join(interviews.groupby('Morality Origin').median().sort_values(by='Shift').index.tolist()))    

#Plot silhouette score for each wave
def plot_silhouette_score(interviews):
    #Compute cosine distance between all pairs of interviews
    interviews = interviews.reset_index(names='id')
    interviews = interviews.merge(interviews, how='cross', suffixes=('', '_'))
    interviews['Distance'] = interviews.apply(lambda x: cosine(x['Embeddings'], x['Embeddings_']), axis=1)
    interviews = interviews.drop(columns=['Embeddings', 'Embeddings_', 'id_'])

    #Average for each pair of interview-wave
    interviews = interviews.groupby(['id', 'Name', 'Name_']).mean().reset_index()
    interviews = interviews.pivot(index=['id', 'Name'], columns='Name_', values='Distance')
    interviews = interviews.reset_index().rename_axis(None, axis=1).drop(columns=['id'])

    #Compute Silhouette score
    interviews['b'] = interviews.apply(lambda i: min([i[w] for w in set(['Wave 1', 'Wave 2', 'Wave 3']) - set([i['Name']])]), axis=1)
    interviews['a'] = interviews.apply(lambda i: i[i['Name']], axis=1)
    interviews['Silhouette Score'] = (interviews['b'] - interviews['a']) / interviews[['a', 'b']].max(axis=1)
    interviews = interviews.sort_values(by='Name')

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 5))
    ax = sns.boxplot(data=interviews, y='Name', x='Silhouette Score', orient='h', palette='Set2')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Silhouette Score')
    plt.savefig('data/plots/morality-silhouette_score.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #Hyperparameters
    config = [1,2,3,4]
    model = 'lg'

    embeddings_file = 'data/cache/morality_embeddings_'+model+'.pkl'
    moral_foundations_file = 'data/cache/moral_foundations_'+model+'.pkl'
    transformation_matrix_file = 'data/cache/transformation_matrix_'+model+'.pkl'
    
    temporal_embeddings_file = 'data/cache/temporal_morality_embeddings_'+model+'.pkl'

    dim_reduction = 'TSNE'
    perplexity = 5

    #Load data
    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[['Wave', 'R:Morality_Embeddings']].dropna().rename(columns={'Wave': 'Name', 'R:Morality_Embeddings': 'Embeddings'})
    interviews['Name'] = interviews['Name'].apply(lambda x: 'Wave ' + str(x))
    moral_foundations = pd.read_pickle(moral_foundations_file)
    temporal_interviews = pd.read_pickle(temporal_embeddings_file)
    
    #Τransform embeddings
    interviews['Embeddings'] = transform_embeddings(interviews['Embeddings'], transformation_matrix_file)
    moral_foundations['Embeddings'] = transform_embeddings(moral_foundations['Embeddings'], transformation_matrix_file)
    for wave in ['Wave 1', 'Wave 2', 'Wave 3']:
        temporal_interviews[wave+':R:Morality_Embeddings'] = transform_embeddings(temporal_interviews[wave+':R:Morality_Embeddings'], transformation_matrix_file)

    for c in config:
        if c == 1:
            plot_morality_embeddings(interviews, dim_reduction, perplexity)
        elif c == 2:
            plot_moral_foundations(interviews, moral_foundations)
        elif c == 3:
            plot_silhouette_score(interviews)
        elif c == 4:
            plot_semantic_shift(temporal_interviews)