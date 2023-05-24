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

from constants import INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS
from transcript_parser import wave_parser


#Compute embeddings for a folder of transcripts
def compute_embeddings(wave_folder, output_file, section_list=None):
    pandarallel.initialize()

    nlp = spacy.load("en_core_web_trf")
    interviews = wave_parser(wave_folder)

    if section_list:
        for section in section_list:
            interviews[section + ' Embeddings'] = interviews[section].parallel_apply(lambda x: nlp(x).vector if not pd.isna(x) else pd.NA)
    else:
        for section in INTERVIEW_SECTIONS:
            for participant in INTERVIEW_PARTICIPANTS:
                interviews[participant + ' ' + section + ' Embeddings'] = interviews[participant + ' ' + section].parallel_apply(lambda x: nlp(x).vector if not pd.isna(x) else pd.NA)

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
def plot_embeddings(embeddings_file, perplexity=30, k=3):

    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[['Name of Interviewer', 'I: Morality Embeddings']].dropna()

    interviewers = interviews.groupby('Name of Interviewer').mean()

    interviewers_indices = k_most_distant_interviewers(interviewers['I: Morality Embeddings'], k)
    selected_interviewers = interviewers.iloc[interviewers_indices].index.tolist()

    embeddings = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(interviews['I: Morality Embeddings'].apply(pd.Series))
    #embeddings = PCA(n_components=2, whiten=True, random_state=42).fit_transform(interviews['I: Morality Embeddings'].apply(pd.Series))

    interviews = interviews[['Name of Interviewer']].join(pd.DataFrame(embeddings))

    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(20, 20))

    data = interviews[interviews['Name of Interviewer'].isin(selected_interviewers)]
    ax = sns.kdeplot(data=data, x=0, y=1, hue='Name of Interviewer', fill=True, alpha=0.5)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('Interviewer Embeddings')
    plt.show()




if __name__ == '__main__':
    compute_embeddings('downloads/wave_1', 'outputs/wave_1_embeddings.pkl', ['I: Morality'])
    for k in [2, 3, 4, 5]:
        for perplexity in [5]:
            print('k =', k, 'perplexity =', perplexity)
            plot_embeddings('outputs/wave_1_embeddings.pkl', perplexity, k)

