import os
import re

import numpy as np
import openai
import pandas as pd
import spacy
import torch
from __init__ import *
from pandarallel import pandarallel
from sklearn.linear_model import LinearRegression, Ridge
from torch.nn.functional import cosine_similarity
from transformers import BartModel, BartTokenizer, BertModel, BertTokenizer, pipeline

from src.helpers import CHATGPT_PROMPT, CODERS, MERGE_MORALITY_ORIGINS, MORALITY_ESTIMATORS, MORALITY_ORIGIN, MORALITY_ORIGIN_EXPLAINED, NEWLINE, UNCERTAINT_TERMS
from src.parser import merge_codings, wave_parser


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

    return transformation_matrix

#Locate morality section in interviews
def locate_morality_section(interviews, section):
    morality_origin_wave_1 = interviews[interviews['Wave'].isin([1])]['R:Morality:M4']
    morality_origin_wave_2 = interviews[interviews['Wave'].isin([2])][['R:Morality:M2', 'R:Morality:M4', 'R:Morality:M6']].apply(lambda l: NEWLINE.join([t for t in l if not pd.isna(t)]), axis=1).replace('', np.nan)
    morality_origin_wave_3 = interviews[interviews['Wave'].isin([3])][['R:Morality:M2', 'R:Morality:M5', 'R:Morality:M7']].apply(lambda l: NEWLINE.join([t for t in l if not pd.isna(t)]), axis=1).replace('', np.nan)
    morality_origin = pd.concat([morality_origin_wave_1, morality_origin_wave_2, morality_origin_wave_3])
    interviews = interviews.join(morality_origin.rename(section))
    interviews = interviews.dropna(subset=[section]).reset_index(drop=True)
    return interviews

#Overfit model to codings
def inform_morality_origin_model(interviews):
    #Compute golden labels
    codings = merge_codings(interviews)
    golden_labels = codings[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]]
    golden_labels.columns = MORALITY_ORIGIN

    #Compute coefficients for more accurate morality origin estimation
    coefs = {}
    for mo in MORALITY_ORIGIN:
        regr = LinearRegression(fit_intercept=False)
        regr.fit(codings[mo].values.reshape(-1, 1), golden_labels[mo].values.reshape(-1, 1))
        coefs[mo] = regr.coef_[0][0]
    coefs = pd.Series(coefs)

    #Multiply with coefficients and add random gaussian noise
    interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN] * coefs
    interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN].clip(lower=0.0, upper=1.0)
    interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN] - pd.DataFrame(abs(np.random.default_rng(42).normal(0, 1e-1, interviews[MORALITY_ORIGIN].shape)), columns=MORALITY_ORIGIN) * (interviews[MORALITY_ORIGIN] > .99).astype(int)

    return interviews

#Compute morality origin of interviews
def compute_morality_origin_model(interviews, model, section, dictionary_file='data/misc/eMFD.pkl'):
    #Zero-shot model
    if model in ['entail', 'entail_ml', 'entail_explained', 'entail_ml_explained']:
        #Premise and hypothesis templates
        hypothesis_template = 'This example is {}.'
        morality_pipeline =  pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

        #Model variants
        multi_label = True if model in ['entail_ml', 'entail_ml_explained'] else False
        morality_dictionary = MORALITY_ORIGIN_EXPLAINED if model in ['entail_explained', 'entail_ml_explained'] else {mo:mo for mo in MORALITY_ORIGIN}

        #Trasformation functions
        classifier = lambda text: morality_pipeline(text.split(NEWLINE), list(morality_dictionary.keys()), hypothesis_template=hypothesis_template, multi_label=multi_label)
        aggregator = lambda l: pd.DataFrame([{morality_dictionary[l]:s for l, s in zip(r['labels'], r['scores'])} for r in l]).mean()
        full_pipeline = lambda text: aggregator(classifier(text))

        #Classify morality origin and join results
        morality_origin = interviews[section].apply(full_pipeline)
        interviews = interviews.join(morality_origin)

        #Normalize scores
        interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN].div(interviews[MORALITY_ORIGIN].sum(axis=1), axis=0).fillna(0.0)

    #ChatGPT model
    elif model == 'chatgpt':
        #Call OpenAI API
        openai.api_key = os.getenv('OPENAI_API_KEY')
        tokenizer = lambda text, token_limit=128: ' '.join(text.split(' ')[:token_limit])
        classifier = lambda text: openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role': 'system', 'content': CHATGPT_PROMPT},{'role': 'user','content': text}], temperature=0, max_tokens=64, top_p=1, frequency_penalty=0, presence_penalty=0)
        aggregator = lambda response: pd.Series({mo:float(re.search(mo+'.*:(.*?)(\n|$)', re.sub('100%', '1.0', response['choices'][0]['message']['content'])).group(1).strip()) if re.search(mo+'.*:(.*?)(\n|$)', response['choices'][0]['message']['content']) else 0.0 for mo in MORALITY_ORIGIN})
        full_pipeline = lambda text: aggregator(classifier(tokenizer(text)))

        #Classify morality origin and join results
        morality_origin = interviews[section].apply(full_pipeline)
        interviews = interviews.join(morality_origin)

        #Normalize scores
        interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN].div(interviews[MORALITY_ORIGIN].sum(axis=1), axis=0).fillna(0.0)

    #Embeddings models
    else:
        #Compute embeddings
        vectorizer = get_vectorizer(model=model, parallel=True)
        nlp = spacy.load('en_core_web_lg')
        interviews = interviews[~interviews[section].isna()]
        interviews[section + '_Embeddings'] = vectorizer(interviews[section])
        
        #Transform embeddings
        transformation_matrix = embed_eMFD(dictionary_file, model)
        interviews[section + '_Embeddings'] = interviews[section + '_Embeddings'].apply(pd.Series).apply(lambda x: np.dot(x, transformation_matrix.T), axis=1)

        #Compute cosine similarity with morality origin vectors
        morality_origin = pd.Series({mo:nlp(mo).vector for mo in MORALITY_ORIGIN})
        for mo in MORALITY_ORIGIN:
            interviews[mo] = interviews[section + '_Embeddings'].apply(lambda e: cosine_similarity(torch.from_numpy(e).view(1, -1), torch.from_numpy(morality_origin[mo]).view(1, -1)).numpy()[0])
        
        #Normalize similarity scores
        interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN].apply(lambda x: pd.Series({mo:p for mo, p in zip(MORALITY_ORIGIN, torch.nn.functional.softmax(torch.from_numpy(x.to_numpy()), dim=0).numpy())}), axis=1)
    
    return interviews

#Count words in morality section
def count_words(interviews, section):
    nlp = spacy.load('en_core_web_lg')
    count = lambda section : 0 if pd.isna(section) else sum([1 for token in nlp(section) if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']])
    interviews[section + '_Word_Count'] = interviews[section].map(count)
    interviews = interviews[interviews[section + '_Word_Count'] < interviews[section + '_Word_Count'].quantile(.95)].reset_index(drop=True)
    return interviews

#Count uncertain terms in morality section
def count_uncertain_terms(interviews, section):
    pattern = r'\b(' + '|'.join(re.escape(term) for term in UNCERTAINT_TERMS) + r')\b'
    count = lambda section : 0 if pd.isna(section) else len(re.findall(pattern, section.lower()))
    interviews[section + '_Uncertain_Terms'] = interviews[section].map(count)
    return interviews


if __name__ == '__main__':
    #Hyperparameters
    config = [3]
    models = ['lg', 'bert', 'bart', 'chatgpt', 'entail', 'entail_explained', 'entail_ml', 'entail_ml_explained']
    section = 'Morality_Origin'

    for c in config:
        if c == 1:
            display_notification = lambda notification: os.system("osascript -e 'display notification \"\" with title \""+notification+"\"'")            
            for model in models:
                interviews = wave_parser(morality_breakdown=True)
                interviews = locate_morality_section(interviews, section)
                interviews = compute_morality_origin_model(interviews, model, section)
                interviews.to_pickle('data/cache/morality_model-'+model+'.pkl')
                display_notification(model + ' Morality Origin Computed!')
        elif c == 2:
            interviews = pd.read_pickle('data/cache/morality_model-entail_ml_explained.pkl')
            if MERGE_MORALITY_ORIGINS:
                interviews['Intuitive'] = interviews['Experience']
                interviews['Consequentialist'] = interviews['Consequences']
                interviews['Social'] = interviews[['Family', 'Community', 'Friends']].max(axis=1)
                interviews['Theistic'] = interviews['Holy Scripture']
                interviews = interviews.drop(['Experience', 'Consequences', 'Family', 'Community', 'Friends', 'Media', 'Laws', 'Holy Scripture'], axis=1)
            interviews = inform_morality_origin_model(interviews)
            interviews.to_pickle('data/cache/morality_model-top.pkl')
        elif c == 3:
            interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
            interviews = count_words(interviews, section)
            interviews = count_uncertain_terms(interviews, section)
            interviews.to_pickle('data/cache/morality_model-top.pkl')