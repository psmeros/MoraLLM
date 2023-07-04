import numpy as np
import pandas as pd
import spacy
from __init__ import *
from pandarallel import pandarallel
from simpletransformers.language_representation import RepresentationModel
from sklearn.linear_model import LinearRegression

from preprocessing.transcript_parser import wave_parser

def get_vectorizer(model='lg', parallel=False):
    if model == 'trf':
        transformer = RepresentationModel(model_type='bert', model_name='bert-base-uncased', use_cuda=False)
        vectorizer = lambda x: pd.Series(transformer.encode_sentences(x, combine_strategy='mean').tolist())
    elif model in ['lg', 'md']:
        nlp = spacy.load('en_core_web_'+model)
        if parallel:
            pandarallel.initialize()
            vectorizer = lambda x: x.parallel_apply(lambda y: nlp(y).vector)
        else:
            vectorizer = lambda x: x.apply(lambda y: nlp(y).vector)
    return vectorizer

#Compute embeddings for a folder of transcripts
def compute_embeddings(wave_folder, output_file, section, morality_breakdown=False, keep_POS=True, model='lg'):
    interviews = wave_parser(wave_folder, morality_breakdown=morality_breakdown)

    #Keep only POS of interest
    if keep_POS:
        nlp = spacy.load('en_core_web_lg')
        pandarallel.initialize()
        interviews[section] = interviews[section].parallel_apply(lambda s: ' '.join(set([w.text for w in nlp(s.lower()) if w.pos_ in ['NOUN', 'ADJ']])).strip() if not pd.isna(s) else pd.NA)

    #Compute embeddings
    interviews = interviews.dropna(subset=[section])
    vectorizer = get_vectorizer(model=model, parallel=True)
    interviews[section + '_Embeddings'] = vectorizer(interviews[section])
    
    #Drop interviews with no embeddings
    interviews = interviews.dropna(subset=[section + '_Embeddings'])
    interviews = interviews[interviews[section + '_Embeddings'].apply(lambda x: sum(x) != 0)]

    interviews.to_pickle(output_file)

#Transform embeddings to match anchor embeddings
def transform_embeddings(embeddings, moral_foundations, model='lg'):

    #Average Vice and Virtue embeddings
    moral_foundations['Name'] = moral_foundations['Name'].apply(lambda x: x.split('.')[0].capitalize())
    moral_foundations = moral_foundations.groupby('Name').mean().reset_index()

    #Compute old embeddings
    vectorizer = get_vectorizer(model=model)
    moral_foundations['Old Embeddings'] = vectorizer(moral_foundations['Name'].str.lower())

    #Find transformation matrix
    regressor = LinearRegression()
    regressor.fit(moral_foundations['Old Embeddings'].apply(pd.Series), moral_foundations['Embeddings'].apply(pd.Series))
    transformation_matrix = regressor.coef_.T

    #Transform embeddings
    embeddings = embeddings.apply(pd.Series).apply(lambda x: np.dot(x, transformation_matrix), axis=1)

    return embeddings

#Compute morality anchors based on morality dictionary
def compute_eMFD_embeddings(dictionary_file, output_file):
    dictionary = pd.DataFrame(pd.read_pickle(dictionary_file)).T
    dictionary = dictionary.reset_index(names=['word'])

    vectorizer = get_vectorizer()
    dictionary['Embeddings'] = vectorizer(dictionary['word'].str.lower())

    moral_foundations = pd.DataFrame()

    for column in dictionary.columns:
        if column not in ['word', 'Embeddings']:
            moral_foundations[column] = sum(dictionary['Embeddings']*dictionary[column])/sum(dictionary[column])

    moral_foundations = moral_foundations.T
    moral_foundations['Embeddings'] = moral_foundations.apply(lambda x: np.array(x), axis=1)
    moral_foundations = moral_foundations[['Embeddings']]
    moral_foundations = moral_foundations.reset_index(names=['Name'])

    moral_foundations.to_pickle(output_file)


if __name__ == '__main__':
    wave_folder = 'data/waves'
    model = 'lg'
    section = 'R:Morality'
    output_file = 'data/cache/morality_embeddings_POS_'+model+'.pkl'
    compute_embeddings(wave_folder=wave_folder, output_file=output_file, model=model, section=section)

    dictionary_file = 'data/misc/eMFD.pkl'
    output_file = 'data/cache/moral_foundations.pkl'
    compute_eMFD_embeddings(dictionary_file=dictionary_file, output_file=output_file)
