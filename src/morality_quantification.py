import os
import re

import numpy as np
import openai
import pandas as pd
from sklearn.preprocessing import minmax_scale
import spacy
import torch
from textstat.textstat import textstat
from torch.nn.functional import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from __init__ import *
from src.helpers import CHATGPT_PROMPT, MORALITY_ORIGIN, MORALITY_ORIGIN_EXPLAINED, NEWLINE, UNCERTAINT_TERMS
from src.parser import wave_parser


#Compute morality source of interviews
def compute_morality_source(models):
    interviews = wave_parser()

    #Locate morality text in interviews
    morality_text = 'Morality Text'
    interviews[morality_text] = ''
    interviews.loc[interviews['Wave'] == 1, morality_text] = interviews.loc[interviews['Wave'] == 1].apply(lambda i: i['R:Morality:M4'], axis=1)
    interviews.loc[interviews['Wave'] == 2, morality_text] = interviews.loc[interviews['Wave'] == 2].apply(lambda i: NEWLINE.join([t for t in [i['R:Morality:M2'], i['R:Morality:M4'], i['R:Morality:M6']] if not pd.isna(t)]), axis=1)
    interviews.loc[interviews['Wave'] == 3, morality_text] = interviews.loc[interviews['Wave'] == 3].apply(lambda i: NEWLINE.join([t for t in [i['R:Morality:M2'], i['R:Morality:M5'], i['R:Morality:M7']] if not pd.isna(t)]), axis=1)
    interviews[morality_text] = interviews[morality_text].replace('', np.nan)
    interviews = interviews.dropna(subset=[morality_text]).reset_index(drop=True)

    #Compute all models
    for model in models:
        data = interviews.copy()

        #NLI model
        if model in ['entail', 'entail_ml', 'entail_explained', 'entail_ml_explained']:
            #Premise and hypothesis templates
            hypothesis_template = 'This example is {}.'
            model_params = {'device':0} if torch.cuda.is_available() else {}
            morality_pipeline = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', **model_params)

            #Model variants
            multi_label = True if model in ['entail_ml', 'entail_ml_explained'] else False
            morality_dictionary = MORALITY_ORIGIN_EXPLAINED if model in ['entail_explained', 'entail_ml_explained'] else {mo:mo for mo in MORALITY_ORIGIN}

            #Trasformation functions
            classifier = lambda text: morality_pipeline(text.split(NEWLINE), list(morality_dictionary.keys()), hypothesis_template=hypothesis_template, multi_label=multi_label)
            aggregator = lambda l: pd.DataFrame([{morality_dictionary[l]:s for l, s in zip(r['labels'], r['scores'])} for r in l]).mean()
            full_pipeline = lambda text: aggregator(classifier(text))

            #Classify morality origin and join results
            morality_origin = data[morality_text].apply(full_pipeline)
            data = data.join(morality_origin)

        #ChatGPT model
        elif model == 'chatgpt':
            #Call OpenAI API
            openai.api_key = os.getenv('OPENAI_API_KEY')
            tokenizer = lambda text, token_limit=128: ' '.join(text.split(' ')[:token_limit])
            classifier = lambda text: openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role': 'system', 'content': CHATGPT_PROMPT},{'role': 'user','content': text}], temperature=.2, max_tokens=32, frequency_penalty=0, presence_penalty=0)
            aggregator = lambda response: pd.Series({mo:float(re.search(mo+'.*:(.*?)(\n|$)', response['choices'][0]['message']['content']).group(1).strip()) if re.search(mo+'.*:(.*?)(\n|$)', response['choices'][0]['message']['content']) else 0.0 for mo in MORALITY_ORIGIN})
            full_pipeline = lambda text: aggregator(classifier(tokenizer(text)))

            #Classify morality origin and join results
            morality_origin = data[morality_text].apply(full_pipeline)
            data = data.join(morality_origin)

        #SBERT model
        elif model == 'sbert':
            #Compute embeddings
            nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
            vectors = pd.DataFrame(nlp_model.encode(data[morality_text])).apply(np.array, axis=1)

            #Compute cosine similarity with morality origin vectors
            morality_origin = pd.Series({mo:nlp_model.encode(mo) for mo in MORALITY_ORIGIN})    
            data[MORALITY_ORIGIN] = pd.DataFrame([vectors.apply(lambda e: cosine_similarity(torch.from_numpy(e).view(1, -1), torch.from_numpy(morality_origin[mo]).view(1, -1)).numpy()[0]) for mo in MORALITY_ORIGIN], index=MORALITY_ORIGIN).T

        #SpaCy model
        elif model == 'lg':
            #Compute embeddings
            nlp_model = spacy.load('en_core_web_lg')
            vectors = data[morality_text].apply(lambda s: np.mean([w.vector for w in nlp_model(s) if w.pos_ in ['NOUN', 'ADJ', 'VERB']], axis=0) if not pd.isna(s) else s)

            #Compute cosine similarity with morality origin vectors
            morality_origin = pd.Series({mo:nlp_model(mo).vector for mo in MORALITY_ORIGIN})    
            data[MORALITY_ORIGIN] = pd.DataFrame([vectors.apply(lambda e: cosine_similarity(torch.from_numpy(e).view(1, -1), torch.from_numpy(morality_origin[mo]).view(1, -1)).numpy()[0]) for mo in MORALITY_ORIGIN], index=MORALITY_ORIGIN).T

        data.to_pickle('data/cache/morality_model-' + model + '.pkl')

def compute_linguistics(model):
    data = pd.read_pickle('data/cache/morality_model-' + model + '.pkl')
    morality_text = 'Morality Text'

    #Count words in morality text
    nlp = spacy.load('en_core_web_lg')
    count = lambda section : 0 if pd.isna(section) else sum([1 for token in nlp(section) if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']])
    data['Verbosity'] = data[morality_text].map(count)
    data = data[data['Verbosity'] < data['Verbosity'].quantile(.95)].reset_index(drop=True)
    
    #Count uncertain terms in morality text
    pattern = r'\b(' + '|'.join(re.escape(term) for term in UNCERTAINT_TERMS) + r')\b'
    count = lambda section : 0 if pd.isna(section) else len(re.findall(pattern, section.lower()))
    data['Uncertainty'] = data[morality_text].map(count)

    #Measure readability in morality text
    measure = lambda section : 0 if pd.isna(section) else textstat.flesch_reading_ease(section)
    data['Complexity'] = data[morality_text].map(measure)

    #Measure sentiment in morality text
    model_params = {'device':0} if torch.cuda.is_available() else {}
    sentiment_pipeline = pipeline('sentiment-analysis', **model_params)
    cumpute_score = lambda r : (r['score'] + 1)/2 if r['label'] == 'POSITIVE' else (-r['score'] + 1)/2 if r['label'] == 'NEGATIVE' else .5
    data['Sentiment'] = pd.Series(sentiment_pipeline(data[morality_text].tolist(), truncation=True)).map(cumpute_score)

    #Normalize values
    data['Uncertainty'] = minmax_scale(data['Uncertainty'].astype(int) / data['Verbosity'].astype(int))
    data['Verbosity'] = minmax_scale(np.log(data['Verbosity'].astype(int)))
    data['Complexity'] = minmax_scale((data['Complexity']).astype(float))
    data['Sentiment'] = minmax_scale(data['Sentiment'].astype(float))

    #Save full morality dialogue
    data.loc[data['Wave'] == 1, morality_text] = data.loc[data['Wave'] == 1, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[4]', l) else '' for l in i.split('\n')]))
    data.loc[data['Wave'] == 2, morality_text] = data.loc[data['Wave'] == 2, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[246]', l) else '' for l in i.split('\n')]))
    data.loc[data['Wave'] == 3, morality_text] = data.loc[data['Wave'] == 3, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[257]', l) else '' for l in i.split('\n')]))

    data.to_pickle('data/cache/morality_model-' + model + '.pkl')


if __name__ == '__main__':
    #Hyperparameters
    config = [1,2]

    for c in config:
        if c == 1:
            models = ['lg', 'sbert', 'chatgpt', 'entail_ml']
            compute_morality_source(models)
        elif c == 2:
            model = 'entail_ml'
            compute_linguistics(model)