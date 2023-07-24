import numpy as np
import pandas as pd
import spacy
from __init__ import *
from pandarallel import pandarallel
from simpletransformers.language_representation import RepresentationModel
from sklearn.linear_model import LinearRegression

from preprocessing.metadata_parser import merge_codings, merge_matches
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

#Compute embeddings for interview sections
def compute_embeddings(interviews, section_list, model):
    
    nlp = spacy.load('en_core_web_lg')
    vectorizer = get_vectorizer(model=model, parallel=True)

    for section in section_list:
    
        #Keep only POS of interest
        interviews[section] = interviews[section].parallel_apply(lambda s: ' '.join([w.text for w in nlp(s.lower()) if w.pos_ in ['NOUN', 'ADJ']]).strip() if not pd.isna(s) else pd.NA)
        interviews = interviews.dropna(subset=[section])

        #Compute embeddings
        interviews[section + '_Embeddings'] = vectorizer(interviews[section])
        
        #Drop interviews with no embeddings
        interviews = interviews.dropna(subset=[section + '_Embeddings'])
        interviews = interviews[interviews[section + '_Embeddings'].apply(lambda x: sum(x) != 0)]
    
    return interviews

#Transform input embeddings
def transform_embeddings(embeddings, transformation_matrix_file = 'data/cache/transformation_matrix.pkl'):
    transformation_matrix = pd.read_pickle(transformation_matrix_file).values
    embeddings = embeddings.apply(pd.Series).apply(lambda x: np.dot(x, transformation_matrix), axis=1)
    return embeddings

#Compute eMFD embeddings and transformation matrix
def embed_eMFD(dictionary_file, model, moral_foundations_file, transformation_matrix_file):
    dictionary = pd.DataFrame(pd.read_pickle(dictionary_file)).T
    dictionary = dictionary.reset_index(names=['word'])

    vectorizer = get_vectorizer(model=model)
    dictionary['Embeddings'] = vectorizer(dictionary['word'].str.lower())

    moral_foundations = pd.DataFrame()

    for column in dictionary.columns:
        if column not in ['word', 'Embeddings']:
            moral_foundations[column] = sum(dictionary['Embeddings']*dictionary[column])/sum(dictionary[column])

    moral_foundations = moral_foundations.T
    moral_foundations['Embeddings'] = moral_foundations.apply(lambda x: np.array(x), axis=1)
    moral_foundations = moral_foundations[['Embeddings']]
    moral_foundations = moral_foundations.reset_index(names=['Name'])

    #Average Vice and Virtue embeddings
    moral_foundations['Name'] = moral_foundations['Name'].apply(lambda x: x.split('.')[0].capitalize())
    moral_foundations = moral_foundations.groupby('Name').mean().reset_index()

    #Compute global embeddings
    global_embeddings = vectorizer(moral_foundations['Name'].str.lower())

    #Find transformation matrix
    regressor = LinearRegression()
    regressor.fit(global_embeddings.apply(pd.Series), moral_foundations['Embeddings'].apply(pd.Series))
    transformation_matrix = pd.DataFrame(regressor.coef_.T)

    #Save files
    moral_foundations.to_pickle(moral_foundations_file)
    transformation_matrix.to_pickle(transformation_matrix_file)


if __name__ == '__main__':
    config = [2]

    if 1 in config:
        interviews = wave_parser()
        model = 'lg'
        section_list = ['R:Morality']
        interviews = compute_embeddings(interviews, section_list, model)
        interviews.to_pickle('data/cache/morality_embeddings_'+model+'.pkl')

    if 2 in config:
        dictionary_file = 'data/misc/eMFD.pkl'
        model = 'lg'
        moral_foundations_file = 'data/cache/moral_foundations.pkl'
        transformation_matrix_file = 'data/cache/transformation_matrix.pkl'
        embed_eMFD(dictionary_file, model, moral_foundations_file, transformation_matrix_file)

    if 3 in config:
        interviews = wave_parser()
        interviews = merge_matches(interviews, wave_list = ['Wave 1', 'Wave 2', 'Wave 3'])
        interviews = merge_codings(interviews)
        model = 'lg'
        section_list = ['Wave 1:R:Morality', 'Wave 2:R:Morality', 'Wave 3:R:Morality']
        interviews = compute_embeddings(interviews, section_list, model)
        interviews.to_pickle('data/cache/temporal_morality_embeddings_'+model+'.pkl')
