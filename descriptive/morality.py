import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.semi_supervised import LabelPropagation
import spacy
from __init__ import *
from simpletransformers.language_representation import RepresentationModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from preprocessing.constants import MORALITY_ENTITIES
from preprocessing.embeddings import compute_embeddings


#Plot morality embeddings of all waves
def plot_morality_embeddings(embeddings_file, model, morality_entities, label_propagation=False, dim_reduction='TSNE', perplexity=5):

    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[['Wave', 'R:Morality_Embeddings']].dropna()
    interviews.columns = ['Name', 'Embeddings']
    interviews['Name'] = interviews['Name'].apply(lambda x: 'Wave ' + str(x))
    
    if model == 'trf':
        transformer = RepresentationModel(model_type='bert', model_name='bert-base-uncased', use_cuda=False)
        vectorizer = lambda x: pd.Series(transformer.encode_sentences(x, combine_strategy='mean').tolist())
    elif model in ['lg', 'md']:
        nlp = spacy.load('en_core_web_lg')
        vectorizer = lambda x: x.apply(lambda y: nlp(y).vector)
    
    morality_entities = pd.DataFrame(morality_entities).melt(var_name='Name', value_name='Embeddings')
    morality_entities['Embeddings'] = vectorizer(morality_entities['Embeddings'].str.lower())
    
    if label_propagation:
        label_propagation = LabelPropagation()
        label_propagation.fit(morality_entities['Embeddings'].apply(pd.Series), morality_entities['Name'])
        interviews['Name'] = interviews['Name'].str.cat(label_propagation.predict(interviews['Embeddings'].apply(pd.Series)), sep=' - ')
        data = interviews
    else:
        data = pd.concat([interviews, morality_entities], ignore_index=True)

    if dim_reduction == 'TSNE':
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        data = data[['Name']].join(pd.DataFrame(tsne.fit_transform(data['Embeddings'].apply(pd.Series))))

    elif dim_reduction == 'PCA':
        pca = PCA(n_components=2, random_state=42)
        pca.fit(morality_entities['Embeddings'].apply(pd.Series))
        data = data[['Name']].join(pd.DataFrame(pca.transform(data['Embeddings'].apply(pd.Series))))

    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))

    ax = sns.scatterplot(data=data, x=0, y=1, hue='Name', palette='Set2', s = 500)

    plt.legend(loc='upper left', markerscale=5, bbox_to_anchor=(1, 1))
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Morality Embeddings')
    plt.gca().get_legend().set_title('')
    plt.savefig('data/plots/morality_embeddings.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    model = 'trf'
    # compute_embeddings('data/waves', 'data/cache/morality_embeddings_'+model+'.pkl', model=model, section='R:Morality')
    plot_morality_embeddings(embeddings_file='data/cache/morality_embeddings_'+model+'.pkl', model=model, morality_entities=MORALITY_ENTITIES, label_propagation=True, dim_reduction='TSNE', perplexity=100)
