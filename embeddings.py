import pandas as pd
import numpy as np
import spacy

from transcript_parser import wave_parser

#Compute embeddings for a folder of transcripts
def compute_embeddings(wave_folder, output_file):
    nlp = spacy.load("en_core_web_sm")
    interviews = wave_parser(wave_folder)

    for person in ['Interviewer', 'Respondent']:
        interviews[person + ' Embeddings'] = interviews[person + ' Full Text'].apply(lambda x: list(nlp(x).vector))

    interviews.to_csv(output_file, index=False)


#Split embeddings into interviewer and respondent
def split_embeddings(embeddings_file):

    wave_1_embeddings = pd.read_csv(embeddings_file)[['Interviewer Embeddings', 'Respondent Embeddings']]
    interviwer_embeddings = wave_1_embeddings['Interviewer Embeddings'].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=',')).apply(pd.Series).to_numpy()
    respondent_embeddings = wave_1_embeddings['Respondent Embeddings'].apply(lambda x: np.fromstring(x[1:-1], dtype=float, sep=',')).apply(pd.Series).to_numpy()

    return interviwer_embeddings, respondent_embeddings
