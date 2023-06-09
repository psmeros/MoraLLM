from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from pandarallel import pandarallel
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from constants import INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS, REFINED_SECTIONS
from transcript_parser import wave_parser
from helpers import display_notification
from simpletransformers.language_representation import RepresentationModel


#Compute embeddings for a folder of transcripts
def compute_embeddings(wave_folder, output_file, section, model='lg'):
                                                   
    interviews = wave_parser(wave_folder)

    #Keep only POS of interest
    nlp = spacy.load('en_core_web_lg')
    interviews[section] = interviews[section].apply(lambda s: ' '.join([w.text for w in nlp(s) if w.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']]) if not pd.isna(s) else pd.NA)
    interviews[section] = interviews[section].str.strip()
    interviews = interviews.dropna(subset=[section])

    #Compute embeddings
    if model == 'trf':
        transformer = RepresentationModel(model_type='bert', model_name='bert-base-uncased', use_cuda=False)
        vectorizer = lambda x: pd.Series(transformer.encode_sentences(x, combine_strategy='mean').tolist())
    elif model in ['lg', 'md']:
        nlp = spacy.load('en_core_web_'+model)
        pandarallel.initialize()
        vectorizer = lambda x: x.parallel_apply(lambda y: nlp(y).vector)

    interviews[section + '_Embeddings'] = vectorizer(interviews[section])
    interviews = interviews.dropna(subset=[section + '_Embeddings'])
    interviews.to_pickle(output_file)

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
def plot_embeddings(embeddings_file, embeddings, dim_reduction='TSNE', perplexity=5, k=3):

    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[['Name of Interviewer', embeddings]].dropna()

    interviewers = interviews.groupby('Name of Interviewer').mean()

    interviewers_indices = k_most_distant_interviewers(interviewers[embeddings], k)
    selected_interviewers = interviewers.iloc[interviewers_indices].index.tolist()


    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))

    if dim_reduction == 'TSNE':
        data = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(interviews[embeddings].apply(pd.Series))
    elif dim_reduction == 'PCA':
        data = PCA(n_components=2, random_state=42).fit_transform(interviews[embeddings].apply(pd.Series))

    data = interviews[['Name of Interviewer']].join(pd.DataFrame(data))
    data = data[data['Name of Interviewer'].isin(selected_interviewers)]
    ax = sns.kdeplot(data=data, x=0, y=1, hue='Name of Interviewer', fill=True, alpha=0.5, bw_adjust=1)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Top ' + str(k) + ' Most Distant Interviewers on '+ embeddings.split()[1])
    plt.gca().get_legend().set_title('Interviewer')
    plt.show()



#Plot morality embeddings of all waves
def plot_morality_embeddings(embeddings_file, model, morality_entities, dim_reduction='TSNE', perplexity=5):

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
    
    data = pd.concat([interviews, morality_entities], ignore_index=True)

    if dim_reduction == 'TSNE':
        data = data[['Name']].join(pd.DataFrame(TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(data['Embeddings'].apply(pd.Series))))
    elif dim_reduction == 'PCA':
        data = data[['Name']].join(pd.DataFrame(PCA(n_components=2, random_state=42).fit_transform(data['Embeddings'].apply(pd.Series))))

    # data = data.groupby('Name').mean().reset_index()

    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))

    # ax = sns.kdeplot(data=data, x=0, y=1, hue='Name', fill=True, alpha=0.5, bw_adjust=1, palette='colorblind')

    ax = sns.scatterplot(data=data, x=0, y=1, hue='Name', palette='Set2', s = 500)

    plt.legend(loc='upper left', markerscale=5, bbox_to_anchor=(1, 1))
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Morality Embeddings')
    plt.gca().get_legend().set_title('')
    plt.show()



if __name__ == '__main__':
    model = 'lg'
    compute_embeddings('data/waves', 'data/cache/morality_embeddings_'+model+'.pkl', model=model, section='R:Morality')
    morality_entities = {'Deontological Morality' : ['Duty', 'Obligation', 'Moral rules', 'Rights', 'Justice', 'Intent', 'Ethical principles', 'Moral absolutes', 'Categorical imperatives', 'Virtue ethics'], 'Consequentialist Morality' : ['Consequences', 'Utility', 'Outcomes', 'Maximizing', 'Well-being', 'Hedonism', 'Utilitarianism', 'Cost-benefit analysis', 'Pragmatism', 'Egoism']}
    plot_morality_embeddings('data/cache/morality_embeddings_'+model+'.pkl', model=model, morality_entities=morality_entities, dim_reduction='TSNE', perplexity=100)
    #plot_embeddings('outputs/wave_1_embeddings.pkl', embeddings='I: Morality Embeddings')
    #plot_embeddings('outputs/wave_1_embeddings.pkl', embeddings='I: Religion Embeddings')
    display_notification('Transformers embeddings computed!')