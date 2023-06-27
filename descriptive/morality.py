import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from __init__ import *
from simpletransformers.language_representation import RepresentationModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation
from scipy.spatial.distance import cosine

from preprocessing.constants import MORALITY_ENTITIES
from preprocessing.embeddings import compute_embeddings, transform_embeddings


#Plot morality embeddings of all waves
def plot_morality_embeddings(embeddings_file, model, anchors, label_propagation=False, dim_reduction='TSNE', perplexity=5):

    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[['Wave', 'R:Morality_Embeddings']].dropna().rename(columns={'Wave': 'Name', 'R:Morality_Embeddings': 'Embeddings'})
    interviews['Name'] = interviews['Name'].apply(lambda x: 'Wave ' + str(x))
    
    interviews['Embeddings'], anchors = transform_embeddings(interviews['Embeddings'], anchors, model)
    
    # morality_entities = morality_entities.groupby('Name').mean().reset_index()
    
    if label_propagation:
        label_propagation = LabelPropagation()
        label_propagation.fit(anchors['Embeddings'].apply(pd.Series), anchors['Name'])
        interviews['Name'] = interviews['Name'].str.cat(label_propagation.predict(interviews['Embeddings'].apply(pd.Series)), sep=' - ')
        data = interviews
    else:
        data = pd.concat([interviews, anchors], ignore_index=True)

    if dim_reduction == 'TSNE':
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        data = data[['Name']].join(pd.DataFrame(tsne.fit_transform(data['Embeddings'].apply(pd.Series))))

    elif dim_reduction == 'PCA':
        pca = PCA(n_components=2, random_state=42)
        pca.fit(anchors['Embeddings'].apply(pd.Series))
        data = data[['Name']].join(pd.DataFrame(pca.transform(data['Embeddings'].apply(pd.Series))))

    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))
    color_palette = sns.color_palette('colorblind')
    
    sns.kdeplot(data=data[data['Name'].isin(['Wave 1', 'Wave 2', 'Wave 3'])], x=0, y=1, hue='Name', fill=True, alpha=0.5, thresh=.4, hue_order=['Wave 1', 'Wave 2', 'Wave 3'], palette=color_palette[:3])
    sns.scatterplot(data=data[~data['Name'].isin(['Wave 1', 'Wave 2', 'Wave 3'])], x=0, y=1, hue='Name', palette=color_palette[3:], s = 500)

    handles, labels = plt.gca().get_legend_handles_labels()
    for p in [patches.Patch(color=color_palette[0], label='Wave 1'), patches.Patch(color=color_palette[1], label='Wave 2'), patches.Patch(color=color_palette[2], label='Wave 3')]:
        handles.append(p)
        labels.append(p.get_label())

    plt.legend(handles, labels, loc='upper left', markerscale=5, bbox_to_anchor=(1, 1), title='')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Morality Embeddings')
    plt.savefig('data/plots/morality_embeddings.png', bbox_inches='tight')
    plt.show()

#Plot moral foundations by wave
def plot_moral_foundations(interviews_file, foundations_file):
    #load data
    interviews = pd.read_pickle(interviews_file)
    interviews = interviews[['Wave', 'R:Morality_Embeddings']].dropna().rename(columns={'Wave': 'Name', 'R:Morality_Embeddings': 'Embeddings'})
    interviews['Name'] = interviews['Name'].apply(lambda x: 'Wave ' + str(x))
    moral_foundations = pd.read_pickle(foundations_file)
    
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
    model = 'lg'
    # compute_embeddings('data/waves', 'data/cache/morality_embeddings_'+model+'.pkl', model=model, section='R:Morality')
    # plot_morality_embeddings(embeddings_file='data/cache/morality_embeddings_'+model+'.pkl', model=model, anchors=MORALITY_ENTITIES, label_propagation=False, dim_reduction='TSNE', perplexity=5)
    plot_moral_foundations(interviews_file='data/cache/morality_embeddings_'+model+'.pkl', foundations_file='data/cache/moral_foundations.pkl')

