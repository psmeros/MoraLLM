import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from __init__ import *
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation

from preprocessing.embeddings import compute_embeddings, transform_embeddings


#Plot morality embeddings of all waves
def plot_morality_embeddings(embeddings_file, moral_foundations_file, model, label_propagation=False, dim_reduction='TSNE', perplexity=5):

    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[['Wave', 'R:Morality_Embeddings']].dropna().rename(columns={'Wave': 'Name', 'R:Morality_Embeddings': 'Embeddings'})
    interviews['Name'] = interviews['Name'].apply(lambda x: 'Wave ' + str(x))
    
    moral_foundations = pd.read_pickle(moral_foundations_file)
    interviews['Embeddings'] = transform_embeddings(interviews['Embeddings'], moral_foundations, model=model)
        
    if label_propagation:
        label_propagation = LabelPropagation()
        label_propagation.fit(moral_foundations['Embeddings'].apply(pd.Series), moral_foundations['Name'])
        interviews['Name'] = interviews['Name'].str.cat(label_propagation.predict(interviews['Embeddings'].apply(pd.Series)), sep=' - ')

    #Dimensionality reduction
    data = interviews
    dim_reduction_alg = TSNE(n_components=2, perplexity=perplexity, random_state=42) if dim_reduction == 'TSNE' else PCA(n_components=2, random_state=42) if dim_reduction == 'PCA' else None
    data = data[['Name']].join(pd.DataFrame(dim_reduction_alg.fit_transform(data['Embeddings'].apply(pd.Series))))
    data = data.dropna()

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))
    color_palette = sns.color_palette('Set2')
    
    waves = ['Wave 1', 'Wave 2', 'Wave 3']
    sns.kdeplot(data=data[data['Name'].isin(waves)], x=0, y=1, hue='Name', fill=True, alpha=0.5, hue_order=waves, palette=color_palette[:3])

    plt.xlabel('')
    plt.ylabel('')
    plt.title('Morality Embeddings')
    plt.savefig('data/plots/morality_embeddings.png', bbox_inches='tight')
    plt.show()

#Plot moral foundations by wave
def plot_moral_foundations(embeddings_file, moral_foundations_file):
    #load data
    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[['Wave', 'R:Morality_Embeddings']].dropna().rename(columns={'Wave': 'Name', 'R:Morality_Embeddings': 'Embeddings'})
    interviews['Name'] = interviews['Name'].apply(lambda x: 'Wave ' + str(x))
    moral_foundations = pd.read_pickle(moral_foundations_file)
    
    #compute similarity
    interviews = interviews.merge(moral_foundations, how='cross')
    interviews['Similarity'] = interviews.apply(lambda x: 1 - cosine(x['Embeddings_x'], x['Embeddings_y']), axis=1)
    interviews = interviews[['Name_x', 'Name_y', 'Similarity']].rename(columns={'Name_x': 'Wave', 'Name_y': 'Anchor'})

    #plot boxplots
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 20))
    ax = sns.boxplot(data=interviews, y='Anchor', x='Similarity', hue='Wave', hue_order=['Wave 1', 'Wave 2', 'Wave 3'], orient='h', whis=[10, 100], palette='Set2')
    ax.legend(title='', loc='upper center', bbox_to_anchor=(0.3, -0.03), ncol=3)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Moral Foundations by Wave')
    plt.savefig('data/plots/moral_foundations.png', bbox_inches='tight')
    plt.show()

    #aggregate similarity
    print(interviews.groupby('Wave').mean(numeric_only=True))
    

if __name__ == '__main__':
    model='lg'
    embeddings_file='data/cache/morality_embeddings_'+model+'.pkl'
    moral_foundations_file='data/cache/moral_foundations.pkl'
    label_propagation=False
    dim_reduction='TSNE'
    perplexity=5
    plot_morality_embeddings(embeddings_file=embeddings_file, moral_foundations_file=moral_foundations_file, model=model, label_propagation=label_propagation, dim_reduction=dim_reduction, perplexity=perplexity)

    model='lg'
    embeddings_file='data/cache/morality_embeddings_'+model+'.pkl'
    moral_foundations_file='data/cache/moral_foundations.pkl'
    plot_moral_foundations(embeddings_file=embeddings_file, moral_foundations_file=moral_foundations_file)

