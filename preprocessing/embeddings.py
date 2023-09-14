import numpy as np
import pandas as pd
import spacy
import torch
from __init__ import *
from pandarallel import pandarallel
from sklearn.linear_model import Ridge
from torch.nn.functional import cosine_similarity
from transformers import BartModel, BartTokenizer, BertModel, BertTokenizer, pipeline

from preprocessing.constants import MORALITY_ORIGIN
from preprocessing.helpers import display_notification
from preprocessing.metadata_parser import merge_codings, merge_matches
from preprocessing.transcript_parser import wave_parser


#Return a SpaCy, BERT, or BART vectorizer
def get_vectorizer(model='lg', parallel=False, filter_POS=True):
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
            input = tokenizer(text, return_tensors='pt')

            #Split the input text into chunks of max_chunk_length
            num_chunks = (input['input_ids'].size(1) - 1) // tokenizer.model_max_length + 1
            chunked_input_ids = torch.chunk(input['input_ids'], num_chunks, dim=1)
            chunked_attention_mask = torch.chunk(input['attention_mask'], num_chunks, dim=1)

            #Initialize an empty tensor to store the embeddings
            all_embeddings = []

            #Forward pass through the model to get the embeddings for each chunk
            with torch.no_grad():
                for (input_ids, attention_mask) in zip(chunked_input_ids, chunked_attention_mask):

                    #Input and Output of the transformer model
                    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                    outputs = model(**inputs, output_attentions=True)

                    #Extract the embeddings from the model's output (max-pooling)
                    # attention_scores = torch.nn.functional.softmax(torch.max(torch.max(outputs.attentions[-1], dim=1).values, dim=1).values, dim=1)
                    # token_embeddings = outputs.last_hidden_state[0]
                    # embeddings = torch.matmul(attention_scores, token_embeddings)
                    embeddings = torch.max(outputs.last_hidden_state[0], dim=0, keepdim=True).values
                    all_embeddings.append(embeddings)

            #Concatenate and aggegate the embeddings from all chunks (max-pooling)
            embeddings = torch.max(torch.cat(all_embeddings, dim=0), dim=0).values.numpy()

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
    embeddings = embeddings.apply(pd.Series).apply(lambda x: np.dot(x, transformation_matrix.T), axis=1)
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
    regressor = Ridge(random_state=42)
    regressor.fit(moral_foundations['Local Embeddings'].apply(pd.Series), moral_foundations['Global Embeddings'].apply(pd.Series))
    transformation_matrix = pd.DataFrame(regressor.coef_)
    moral_foundations= moral_foundations.rename(columns={'Local Embeddings': 'Embeddings'})[['Name', 'Embeddings']]

    return moral_foundations, transformation_matrix

#Compute morality origin of interviews
def compute_morality_origin(embeddings_file, transformation_matrix_file):
    nlp = spacy.load('en_core_web_lg')

    #Load and transform data
    interviews = pd.read_pickle(embeddings_file)
    embeddings = transform_embeddings(interviews['Morality_Origin_Embeddings'], transformation_matrix_file)

    #Compute cosine similarity with morality origin vectors
    morality_origin = pd.Series({mo:nlp(mo).vector for mo in MORALITY_ORIGIN})
    for mo in MORALITY_ORIGIN:
        interviews[mo] = embeddings.apply(lambda e: cosine_similarity(torch.from_numpy(e).view(1, -1), torch.from_numpy(morality_origin[mo]).view(1, -1)).numpy()[0])
    interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN].apply(lambda x: pd.Series({mo:p for mo, p in zip(MORALITY_ORIGIN, torch.nn.functional.softmax(torch.from_numpy(x.to_numpy()), dim=0).numpy())}), axis=1)
    
    return interviews

#Compute zero-shot morality origin of interviews
def zero_shot_classification(interviews):
    #Premise and hypothesis templates
    hypothesis_template = 'The morality origin is {}.'
    morality_pipeline =  pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    result_dict = lambda l: pd.DataFrame([{l:s for l, s in zip(r['labels'], r['scores'])} for r in l])
    morality_origin = result_dict(morality_pipeline(interviews['Morality_Origin'].tolist(), MORALITY_ORIGIN, hypothesis_template=hypothesis_template))

    #Join and filter results
    interviews = interviews.join(morality_origin)
    return interviews

if __name__ == '__main__':
    #Hyperparameters
    config = [4]
    models = ['md', 'lg', 'bert', 'bart', 'entail']

    for model in models:
        for c in config:
            if c == 1:
                interviews = wave_parser(morality_breakdown=True)
                morality_origin = pd.concat([interviews[interviews['Wave'].isin([1,2])]['R:Morality:M4'], interviews[interviews['Wave'].isin([3])]['R:Morality:M5']])
                interviews = interviews.join(morality_origin.rename('Morality_Origin'))
                interviews = interviews.dropna(subset=['Morality_Origin']).reset_index(drop=True)
                if model == 'entail':
                    interviews = zero_shot_classification(interviews)
                else:
                    interviews = compute_embeddings(interviews, ['Morality_Origin'], model)
                interviews.to_pickle('data/cache/morality_embeddings_'+model+'.pkl')
                display_notification(model + ' Morality Embeddings Computed!')

            if c == 2:
                dictionary_file = 'data/misc/eMFD.pkl'
                moral_foundations, transformation_matrix = embed_eMFD(dictionary_file, model)
                moral_foundations.to_pickle('data/cache/moral_foundations_'+model+'.pkl')
                transformation_matrix.to_pickle('data/cache/transformation_matrix_'+model+'.pkl')

            if c == 3:
                embeddings_file = 'data/cache/morality_embeddings_'+model+'.pkl'
                transformation_matrix_file = 'data/cache/transformation_matrix_'+model+'.pkl'
                interviews = compute_morality_origin(embeddings_file, transformation_matrix_file)
                interviews.to_pickle('data/cache/morality_embeddings_'+model+'.pkl')

            if c == 4:
                interviews = pd.read_pickle('data/cache/morality_embeddings_'+model+'.pkl')
                interviews = merge_matches(interviews, wave_list = ['Wave 1', 'Wave 2', 'Wave 3'])
                interviews.to_pickle('data/cache/temporal_morality_embeddings_'+model+'.pkl')
                display_notification(model + ' Temporal Morality Embeddings Computed!')