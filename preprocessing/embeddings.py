import numpy as np
import pandas as pd
import spacy
import torch
from __init__ import *
from pandarallel import pandarallel
from sklearn.linear_model import LinearRegression
from transformers import BartModel, BartTokenizer, BertModel, BertTokenizer

from preprocessing.helpers import display_notification
from preprocessing.metadata_parser import merge_codings, merge_matches
from preprocessing.transcript_parser import wave_parser

#Return a SpaCy, BERT, or BART vectorizer
def get_vectorizer(model='lg', parallel=False, filter_POS=True, batch_size=512):
    if model in ['bert', 'bart']:
        #Load the tokenizer and model
        if model == 'bert':
            model_name = 'bert-base-uncased'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
        elif model == 'bart':
            model_name = 'facebook/bart-large-mnli'
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartModel.from_pretrained(model_name)

        def extract_embeddings(text):
            #Tokenize the input text
            input_ids = tokenizer(text, return_tensors='pt', padding=True).input_ids

            #Split the input text into chunks of max_chunk_length
            num_chunks = (input_ids.size(1) - 1) // batch_size + 1
            chunked_input_ids = torch.chunk(input_ids, num_chunks, dim=1)

            #Initialize an empty tensor to store the embeddings
            all_embeddings = []

            #Forward pass through the model to get the embeddings for each chunk
            with torch.no_grad():
                for chunk_ids in chunked_input_ids:
                    outputs = model(input_ids=chunk_ids)

                    #Extract the embeddings from the model's output (max-pooling)
                    embeddings = outputs.last_hidden_state.max(dim=1)

                    #Append the embeddings to the list
                    all_embeddings.append(embeddings)

            #Concatenate and max-pool the embeddings from all chunks
            embeddings = torch.cat(all_embeddings, dim=0).max(dim=0).numpy()

            return embeddings
    
        vectorizer = lambda x: x.apply(extract_embeddings)

    elif model in ['lg', 'md']:
        nlp = spacy.load('en_core_web_'+model)
        if parallel:
            pandarallel.initialize()
        validate_POS = lambda w: w.pos_ in ['NOUN', 'ADJ', 'VERB'] if filter_POS else True
        mean_word_vectors = lambda s: np.mean([w.vector for w in nlp(s) if validate_POS(w)], axis=0)
        vectorizer = lambda x: x.parallel_apply(mean_word_vectors) if parallel else x.apply(mean_word_vectors)
 
    return vectorizer

#Compute embeddings for interview sections
def compute_embeddings(interviews, section_list, model):
    vectorizer = get_vectorizer(model=model, parallel=True)

    for section in section_list:
        #Compute embeddings
        interviews = interviews[~interviews[section].isna()]
        interviews[section + '_Embeddings'] = vectorizer(interviews[section])
        
        #Drop interviews with no embeddings
        interviews = interviews.dropna(subset=[section + '_Embeddings'])
        interviews = interviews[interviews[section + '_Embeddings'].apply(lambda x: sum(x) != 0)]
    
    return interviews

#Transform input embeddings
def transform_embeddings(embeddings, transformation_matrix_file):
    transformation_matrix = pd.read_pickle(transformation_matrix_file).values
    embeddings = embeddings.apply(pd.Series).apply(lambda x: np.dot(x, transformation_matrix), axis=1)
    return embeddings

#Compute eMFD embeddings and transformation matrix
def embed_eMFD(dictionary_file, model):
    #Load data
    dictionary = pd.DataFrame(pd.read_pickle(dictionary_file)).T
    dictionary = dictionary.reset_index(names=['word'])

    #Compute global embeddings
    vectorizer = get_vectorizer(model='lg', parallel=False, filter_POS=False)
    dictionary['Embeddings'] = vectorizer(dictionary['word'].str.lower())
    dictionary = dictionary.dropna(subset=['Embeddings'])

    moral_foundations = pd.DataFrame()

    for column in dictionary.columns:
        if column not in ['word', 'Embeddings']:
            moral_foundations[column] = sum(dictionary['Embeddings']*dictionary[column])/sum(dictionary[column])

    moral_foundations = moral_foundations.T
    moral_foundations['Global Embeddings'] = moral_foundations.apply(lambda x: np.array(x), axis=1)
    moral_foundations = moral_foundations[['Global Embeddings']]
    moral_foundations = moral_foundations.reset_index(names=['Name'])

    #Average Vice and Virtue embeddings
    moral_foundations['Name'] = moral_foundations['Name'].apply(lambda x: x.split('.')[0].capitalize())
    moral_foundations = moral_foundations.groupby('Name').mean().reset_index()

    #Compute local embeddings
    vectorizer = get_vectorizer(model=model, parallel=False, filter_POS=False)
    moral_foundations['Local Embeddings'] = vectorizer(moral_foundations['Name'].str.lower())

    #Drop empty embeddings
    moral_foundations = moral_foundations[moral_foundations.apply(lambda x: (sum(x['Local Embeddings']) != 0) & (sum(x['Global Embeddings']) != 0), axis=1)]

    #Find transformation matrix
    regressor = LinearRegression()
    regressor.fit(moral_foundations['Global Embeddings'].apply(pd.Series), moral_foundations['Local Embeddings'].apply(pd.Series))
    transformation_matrix = pd.DataFrame(regressor.coef_)
    moral_foundations= moral_foundations.rename(columns={'Local Embeddings': 'Embeddings'})[['Name', 'Embeddings']]

    return moral_foundations, transformation_matrix


if __name__ == '__main__':
    config = [1,2,3]
    model = 'bart'

    for c in config:
        if c == 1:
            interviews = wave_parser()
            section_list = ['R:Morality']
            interviews = compute_embeddings(interviews, section_list, model)
            interviews.to_pickle('data/cache/morality_embeddings_'+model+'.pkl')
            display_notification('Morality Embeddings Computed!')

        if c == 2:
            dictionary_file = 'data/misc/eMFD.pkl'
            moral_foundations, transformation_matrix = embed_eMFD(dictionary_file, model)
            moral_foundations.to_pickle('data/cache/moral_foundations_'+model+'.pkl')
            transformation_matrix.to_pickle('data/cache/transformation_matrix_'+model+'.pkl')

        if c == 3:
            interviews = wave_parser()
            interviews = merge_matches(interviews, wave_list = ['Wave 1', 'Wave 2', 'Wave 3'])
            interviews = merge_codings(interviews)
            section_list = ['Wave 1:R:Morality', 'Wave 2:R:Morality', 'Wave 3:R:Morality']
            interviews = compute_embeddings(interviews, section_list, model)
            interviews.to_pickle('data/cache/temporal_morality_embeddings_'+model+'.pkl')
            display_notification('Temporal Morality Embeddings Computed!')
