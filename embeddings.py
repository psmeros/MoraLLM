import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from transcript_parser import wave_parser

#Compute embeddings for a folder of transcripts
def compute_embeddings(wave_folder, output_file):
    nlp = spacy.load("en_core_web_sm")
    interviews = wave_parser(wave_folder)

    for person in ['Interviewer', 'Respondent']:
        interviews[person + ' Embeddings'] = interviews[person + ' Full Text'].apply(lambda x: list(nlp(x).vector))

    interviews.to_csv(output_file, index=False)


#Plot embeddings of a folder of transcripts
def plot_embeddings(embeddings_file):

    wave_1_embeddings = pd.read_csv(embeddings_file)[['Interviewer Embeddings', 'Respondent Embeddings']]

    #Convert string to numpy array
    interviwer_embeddings = wave_1_embeddings['Interviewer Embeddings'].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=',')).apply(pd.Series).to_numpy()
    respondent_embeddings = wave_1_embeddings['Respondent Embeddings'].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=',')).apply(pd.Series).to_numpy()
    
    #Remove NaNs
    interviwer_embeddings = interviwer_embeddings[~np.isnan(interviwer_embeddings).any(axis=1)]
    respondent_embeddings = respondent_embeddings[~np.isnan(respondent_embeddings).any(axis=1)]

    # Use t-SNE to project the embeddings onto a 2D plane
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=100)
    interviwer_embeddings_2d = tsne.fit_transform(interviwer_embeddings)
    respondent_embeddings_2d = tsne.fit_transform(respondent_embeddings)

    # Plot the results
    plt.scatter(interviwer_embeddings_2d[:, 0], interviwer_embeddings_2d[:, 1], c='r')
    plt.scatter(respondent_embeddings_2d[:, 0], respondent_embeddings_2d[:, 1], c='b')
    plt.show()


if __name__ == '__main__':
    plot_embeddings('outputs/wave_1_embeddings.csv')