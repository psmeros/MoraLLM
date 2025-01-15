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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from __init__ import *
from src.helpers import chatgpt_prompt, chatgpt_synthetic_prompt, MORALITY_ORIGIN, MORALITY_ORIGIN_EXPLAINED, UNCERTAINT_TERMS
from src.parser import wave_parser


#Compute morality source of interviews
def compute_morality_source(models, excerpt):
    interviews = pd.read_pickle('data/cache/interviews.pkl')

    #Locate morality text in interviews
    morality_text = 'Morality Text'
    if excerpt == 'full_QnA':
        interviews.loc[interviews['Wave'] == 1, morality_text] = interviews.loc[interviews['Wave'] == 1, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[4]', l) else '' for l in i.split('\n')]) if not pd.isna(i) else '')
        interviews.loc[interviews['Wave'] == 2, morality_text] = interviews.loc[interviews['Wave'] == 2, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[246]', l) else '' for l in i.split('\n')]) if not pd.isna(i) else '')
        interviews.loc[interviews['Wave'] == 3, morality_text] = interviews.loc[interviews['Wave'] == 3, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[257]', l) else '' for l in i.split('\n')]) if not pd.isna(i) else '')
    elif excerpt == 'response':
        interviews.loc[interviews['Wave'] == 1, morality_text] = interviews.loc[interviews['Wave'] == 1].apply(lambda i: i['R:Morality:M4'], axis=1)
        interviews.loc[interviews['Wave'] == 2, morality_text] = interviews.loc[interviews['Wave'] == 2].apply(lambda i: ' '.join([t for t in [i['R:Morality:M2'], i['R:Morality:M4'], i['R:Morality:M6']] if not pd.isna(t)]), axis=1)
        interviews.loc[interviews['Wave'] == 3, morality_text] = interviews.loc[interviews['Wave'] == 3].apply(lambda i: ' '.join([t for t in [i['R:Morality:M2'], i['R:Morality:M5'], i['R:Morality:M7']] if not pd.isna(t)]), axis=1)
    elif excerpt == 'summary':
        interviews[morality_text] = interviews['Morality Summary']
    interviews[morality_text] = interviews[morality_text].replace('', np.nan)
    interviews = interviews.dropna(subset=[morality_text]).reset_index(drop=True)

    #Compute all models
    for model in models:
        data = interviews.copy()

        #NLI model
        if model in ['entail', 'entail_ml', 'entail_explained', 'entail_ml_explained']:
            #Premise and hypothesis templates
            hypothesis_template = 'The reasoning in this example is based on {}.'
            model_params = {'device':0} if torch.cuda.is_available() else {}
            morality_pipeline = pipeline('zero-shot-classification', model='roberta-large-mnli', **model_params)

            #Model variants
            multi_label = True if model in ['entail_ml', 'entail_ml_explained'] else False
            morality_dictionary = MORALITY_ORIGIN_EXPLAINED if model in ['entail_explained', 'entail_ml_explained'] else {mo:mo for mo in MORALITY_ORIGIN}

            #Trasformation functions
            classifier = lambda series: pd.Series(morality_pipeline(series.tolist(), list(morality_dictionary.keys()), hypothesis_template=hypothesis_template, multi_label=multi_label))
            aggregator = lambda r: pd.DataFrame([{morality_dictionary[l]:s for l, s in zip(r['labels'], r['scores'])}]).max()
            
            #Classify morality origin and join results
            morality_origin = classifier(data[morality_text]).apply(aggregator)
            data = data.join(morality_origin)

        #ChatGPT model
        elif model in ['chatgpt_bin', 'chatgpt_quant']:
            response = 'bin' if model == 'chatgpt_bin' else 'quant' if model == 'chatgpt_quant' else ''
            #Call OpenAI API
            openai.api_key = os.getenv('OPENAI_API_KEY')
            classifier = lambda text: [openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role': 'system', 'content': chatgpt_prompt(mo, response)},{'role': 'user','content': text}], temperature=.2, max_tokens=32, frequency_penalty=0, presence_penalty=0, seed=42) for mo in MORALITY_ORIGIN]
            aggregator = lambda r: pd.Series({mo:(lambda n: float(n) if n.strip().isdigit() else 0)(r[i]['choices'][0]['message']['content']) for i, mo in enumerate(MORALITY_ORIGIN)})
            full_pipeline = lambda text: aggregator(classifier(text))

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

        #Seeded LDA model
        elif model == 'lda':
            vectorizer = CountVectorizer(stop_words='english')
            X = vectorizer.fit_transform(data[morality_text]).toarray()
            vocab = vectorizer.get_feature_names_out()

            for mo in MORALITY_ORIGIN:
                    if mo in vocab:
                        X[:, vocab.tolist().index(mo)] += 10

            # Train LDA
            lda = LatentDirichletAllocation(n_components=4, max_iter=100, random_state=42)
            data[MORALITY_ORIGIN] = lda.fit_transform(X)

        data.to_pickle('data/cache/morality_model-' + model + '.pkl')

def compute_linguistics(models):
    #Compute for all models
    for model in models:

        data = pd.read_pickle('data/cache/morality_model-' + model + '.pkl')
        
        #Locate morality text in interviews
        morality_text = 'Morality Text'
        data[morality_text] = ''
        data.loc[data['Wave'] == 1, morality_text] = data.loc[data['Wave'] == 1].apply(lambda i: i['R:Morality:M4'], axis=1)
        data.loc[data['Wave'] == 2, morality_text] = data.loc[data['Wave'] == 2].apply(lambda i: ' '.join([t for t in [i['R:Morality:M2'], i['R:Morality:M4'], i['R:Morality:M6']] if not pd.isna(t)]), axis=1)
        data.loc[data['Wave'] == 3, morality_text] = data.loc[data['Wave'] == 3].apply(lambda i: ' '.join([t for t in [i['R:Morality:M2'], i['R:Morality:M5'], i['R:Morality:M7']] if not pd.isna(t)]), axis=1)
        data[morality_text] = data[morality_text].replace('', np.nan)
        data = data.dropna(subset=[morality_text]).reset_index(drop=True)

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
        sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', **model_params)
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

#Compute synthetic dataset
def compute_synthetic_data(n=25):
    #OpenAI API
    openai.api_key = os.getenv('OPENAI_API_KEY')
    synthesizer = lambda: [openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role': 'system', 'content': chatgpt_synthetic_prompt(mo)},{'role': 'user','content': 'Generate strictly ' + str(n) + ' pairs without enumerating'}], temperature=.2, max_tokens=16384, frequency_penalty=0, presence_penalty=0, seed=42) for mo in MORALITY_ORIGIN]
    aggregator = lambda r: pd.DataFrame(r, index=MORALITY_ORIGIN)['choices'].apply(lambda c: c[0]['message']['content']).str.split('\n').explode().str.split('%').apply(pd.Series).reset_index().dropna().reset_index(drop=True)
    full_pipeline = lambda: aggregator(synthesizer())

    #Generate synthetic data
    data = full_pipeline()
    data.columns = ['Morality', 'Strong Summary', 'Weak Summary']
    data.to_pickle('data/cache/synthetic_data.pkl')

#Compute synthetic morality origin
def compute_synthetic_morality():
    data = pd.read_pickle('data/cache/synthetic_data.pkl')

    #Premise and hypothesis templates
    hypothesis_template = 'The reasoning in this example is based on {}.'
    model_params = {'device':0} if torch.cuda.is_available() else {}
    morality_pipeline = pipeline('zero-shot-classification', model='roberta-large-mnli', **model_params)

    #Trasformation functions
    classifier = lambda series: pd.Series(morality_pipeline(series.tolist(), list(MORALITY_ORIGIN_EXPLAINED.keys()), hypothesis_template=hypothesis_template, multi_label=True))
    aggregator = lambda r: pd.DataFrame([{MORALITY_ORIGIN_EXPLAINED[l]:s for l, s in zip(r['labels'], r['scores'])}]).max()
    
    #Classify morality origin and join results
    morality_origin = classifier(data['Strong Summary']).apply(aggregator)
    data = data.join(morality_origin)
    morality_origin = classifier(data['Weak Summary']).apply(aggregator)
    data = data.join(morality_origin, lsuffix='_strong', rsuffix='_weak')

    data['Distinction'] = data.apply(lambda d: d[d['Morality'] + '_strong'] - d[d['Morality'] + '_weak'], axis=1)
    data = data[['Morality', 'Strong Summary', 'Weak Summary', 'Distinction']]
    data.to_pickle('data/cache/synthetic_data.pkl')

#Binarize continuous morality
def binarize_morality():
    for model in ['nli', 'nli_sum']:
        data = pd.read_pickle('data/cache/morality_model-' + model + '_quant.pkl')
        data[MORALITY_ORIGIN] = (data[MORALITY_ORIGIN] > .5).astype(int)
        data.to_pickle('data/cache/morality_model-' + model + '_bin.pkl')

#Quantize continuous morality
def quantize_morality():
    for model in ['nli', 'nli_sum']:
        data = pd.read_pickle('data/cache/morality_model-' + model + '_quant.pkl')
        data[MORALITY_ORIGIN] = pd.DataFrame([pd.qcut(data[mo], q=5, labels=False) * .25 for mo in MORALITY_ORIGIN]).T
        data.to_pickle('data/cache/morality_model-' + model + '_quant-alt.pkl')

if __name__ == '__main__':
    #Hyperparameters
    config = [4]
    excerpt = 'summary'
    models = ['lda', 'lg', 'sbert', 'chatgpt_bin', 'chatgpt_quant', 'entail_ml', 'entail_ml_explained']

    for c in config:
        if c == 1:
            compute_morality_source(models, excerpt=excerpt)
        elif c == 2:
            compute_linguistics(models)
        elif c == 3:
            compute_synthetic_data()
            compute_synthetic_morality()
        elif c == 4:
            binarize_morality()
            quantize_morality()