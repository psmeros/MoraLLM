import os
import re

import numpy as np
import openai
import pandas as pd
from sklearn.preprocessing import minmax_scale
import spacy
import torch
from torch.nn.functional import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from __init__ import *
from src.helpers import MORALITY_VOCAB, chatgpt_prompt, chatgpt_synthetic_prompt, MORALITY_ORIGIN, MORALITY_ORIGIN_EXPLAINED
from src.parser import clean_morality_tags


#Compute morality source of interviews
def compute_morality_source(models, excerpt):
    interviews = pd.read_pickle('data/cache/interviews.pkl')

    #Locate morality text in interviews
    morality_text = 'Morality Text'
    if excerpt == 'full_QnA':
        interviews.loc[interviews['Wave'] == 1, morality_text] = interviews.loc[interviews['Wave'] == 1, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[4]', l) else '' for l in i.split('\n')]) if not pd.isna(i) else '')
        interviews.loc[interviews['Wave'] == 2, morality_text] = interviews.loc[interviews['Wave'] == 2, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[246]', l) else '' for l in i.split('\n')]) if not pd.isna(i) else '')
        interviews.loc[interviews['Wave'] == 3, morality_text] = interviews.loc[interviews['Wave'] == 3, 'Morality_Full_Text'].apply(lambda i: ''.join([l + '\n' if re.match(r'^[IR]:M[257]', l) else '' for l in i.split('\n')]) if not pd.isna(i) else '')
        interviews[morality_text] = interviews[morality_text].apply(clean_morality_tags)
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

        #Word count model
        elif model == 'wc':
            data[MORALITY_ORIGIN] = data[morality_text].apply(lambda t: pd.Series([sum(1 for w in t.lower().split() if w in MORALITY_VOCAB[mo]) for mo in MORALITY_ORIGIN]) > (len(t.split()) * .01)).astype(int)

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
def binarize_morality(model):
    data = pd.read_pickle('data/cache/morality_model-' + model + '_quant.pkl')
    data[MORALITY_ORIGIN] = (data[MORALITY_ORIGIN].apply(minmax_scale) > .5).astype(int)
    data.to_pickle('data/cache/morality_model-' + model + '_bin.pkl')

#Quantize continuous morality
def quantize_morality(model):
    data = pd.read_pickle('data/cache/morality_model-' + model + '_quant.pkl')
    data[MORALITY_ORIGIN] = pd.DataFrame([pd.qcut(data[mo], q=5, labels=False) * .25 for mo in MORALITY_ORIGIN]).T
    data.to_pickle('data/cache/morality_model-' + model + '_quant-alt.pkl')

if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    excerpt = 'full_QnA'
    models = ['wc', 'lda', 'sbert', 'chatgpt_bin', 'entail_ml_explained']

    for c in config:
        if c == 1:
            compute_morality_source(models, excerpt=excerpt)
        elif c == 2:
            compute_synthetic_data()
            compute_synthetic_morality()
        elif c == 3:
            model = 'sbert'
            binarize_morality(model=model)