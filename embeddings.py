import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from pandarallel import pandarallel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from constants import INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS
from transcript_parser import wave_parser



#Compute embeddings for a folder of transcripts
def compute_embeddings(wave_folder, output_file, section_list=None):
    pandarallel.initialize()

    nlp = spacy.load("en_core_web_lg")
    interviews = wave_parser(wave_folder)

    if section_list:
        for section in section_list:
            interviews[section + ' Embeddings'] = interviews[section].parallel_apply(lambda x: nlp(x).vector if not pd.isna(x) else pd.NA)
    else:
        for section in INTERVIEW_SECTIONS:
            for participant in INTERVIEW_PARTICIPANTS:
                interviews[participant + ' ' + section + ' Embeddings'] = interviews[participant + ' ' + section].parallel_apply(lambda x: nlp(x).vector if not pd.isna(x) else pd.NA)

    interviews.to_pickle(output_file)


#Plot embeddings of a folder of transcripts
def plot_embeddings(embeddings_file):

    interviews = pd.read_pickle(embeddings_file)
    interviews = interviews[['Name of Interviewer', 'I: Morality Embeddings']].dropna()

    embeddings = TSNE(n_components=2, metric="cosine", random_state=42).fit_transform(interviews['I: Morality Embeddings'].apply(pd.Series))
    #embeddings = PCA(n_components=2, whiten=True, random_state=42).fit_transform(interviews['I: Morality Embeddings'].apply(pd.Series))

    interviews = interviews[['Name of Interviewer']].join(pd.DataFrame(embeddings))

    interviewers = interviews['Name of Interviewer'].value_counts()
    interviewers = interviewers[interviewers > 15].index.tolist()

    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(20, 20))

    for selected_interviewers in itertools.combinations(interviewers, 3):
        data = interviews[interviews['Name of Interviewer'].isin(selected_interviewers)]
        ax = sns.kdeplot(data=data, x=0, y=1, hue='Name of Interviewer', fill=True, alpha=0.5)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title('Interviewer Embeddings')
        plt.show()




if __name__ == '__main__':
    #compute_embeddings('downloads/wave_1', 'outputs/wave_1_embeddings.pkl', ['I: Morality'])
    plot_embeddings('outputs/wave_1_embeddings.pkl')



