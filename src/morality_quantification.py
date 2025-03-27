import os
import re
import time

import numpy as np
import openai
import pandas as pd
import requests
from sklearn.preprocessing import minmax_scale
import spacy
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from __init__ import *
from src.helpers import MORALITY_VOCAB, llm_prompt, chatgpt_synthetic_prompt, MORALITY_ORIGIN, MORALITY_ORIGIN_EXPLAINED
from src.parser import clean_morality_tags


#Compute morality source of interviews
def compute_morality_source(models, excerpts):
    for excerpt in excerpts:
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

            #LLM models
            elif model in ['chatgpt_bin', 'deepseek_bin', 'chatgpt_bin_ao', 'deepseek_bin_ao']:
                #Call API
                def call_llm(llm: str, prompt: str, text: str, timeout: int = 15, max_retries: int = 10, backoff_factor: float = 1.0):
                    # Choose model
                    if llm.startswith('deepseek_bin'):
                        url = 'https://openrouter.ai/api/v1/chat/completions'
                        model = 'deepseek/deepseek-chat'
                        api_key = os.getenv('OPENROUTER_API_KEY')
                        temperature = 1.3
                    elif llm.startswith('chatgpt_bin'):
                        url = 'https://api.openai.com/v1/chat/completions'
                        model = 'gpt-4o-mini'
                        api_key = os.getenv('OPENAI_API_KEY')
                        temperature = .2

                    headers = {'Authorization': f'Bearer {api_key}','Content-Type': 'application/json'}
                    data = {'model': model,'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': text}], 'temperature':temperature, 'max_tokens':32, 'seed':42}

                    for attempt in range(max_retries):
                        try:
                            response = requests.post(url, json=data, headers=headers, timeout=timeout)
                            response.raise_for_status()
                            
                            #Parse response
                            response = response.json()
                            parced_response = response['choices'][0]['message']['content'].strip()
                            if llm.endswith('_ao'):
                                parced_response = pd.Series({mo:int(r) for mo, r in zip(MORALITY_ORIGIN, parced_response)})
                                if not parced_response.apply(lambda r: r in [0,1]).all():
                                    raise Exception('Response not parsable')
                            else:
                                parced_response = 0 if '0' in parced_response else 1 if '1' in parced_response else -1
                                if parced_response == -1:
                                    raise Exception('Response not parsable')
                            return parced_response
                        
                        except Exception as e:
                            print(f"Attempt {attempt + 1} failed: {str(e)}")
                            
                            # Handle rate limiting
                            if isinstance(e, requests.exceptions.HTTPError):
                                if e.response.status_code == 429:
                                    retry_after = e.response.headers.get('Retry-After', backoff_factor * (2 ** attempt))
                                    print(f"Rate limited. Retrying after {retry_after} seconds")
                                    time.sleep(float(retry_after))
                                    continue
                                    
                            # Exponential backoff
                            sleep_time = backoff_factor * (2 ** attempt)
                            print(f"Retrying in {sleep_time} seconds...")
                            time.sleep(sleep_time)

                    print('Request failed after max retries')
                    return (pd.Series({mo:-1 for mo in MORALITY_ORIGIN}) if llm.endswith('_ao') else -1)
                 
                full_pipeline = lambda text: (call_llm(llm=model, prompt=llm_prompt('all', 'bin_all'), text=text) if model.endswith('_ao') else pd.Series({mo:call_llm(llm=model, prompt=llm_prompt(mo, 'bin'), text=text) for mo in MORALITY_ORIGIN}))
                tqdm.pandas()

                #Classify morality origin and join results
                morality_origin = data[morality_text].progress_apply(full_pipeline)
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
                nlp_model = spacy.load('en_core_web_lg')
                text = data[morality_text].apply(lambda t: ' '.join([w.lemma_ for w in nlp_model(t) if w.lemma_ in [w for v in MORALITY_VOCAB.values() for w in v]]))
                vectorizer = CountVectorizer(vocabulary=[w for v in MORALITY_VOCAB.values() for w in v])
                lda = LatentDirichletAllocation(n_components=4, max_iter=1000, random_state=42)
                data[MORALITY_ORIGIN] = lda.fit_transform(vectorizer.fit_transform(text))

            #Word count model
            elif model == 'wc':
                nlp_model = spacy.load('en_core_web_lg')
                data[MORALITY_ORIGIN] = data[morality_text].apply(lambda t: pd.Series([sum(1 for w in nlp_model(t) if w.lemma_ in MORALITY_VOCAB[mo]) for mo in MORALITY_ORIGIN]) > 0).astype(int)

            data.to_pickle('data/cache/morality_model-' + model + ('_resp' if excerpt == 'response' else '_sum' if excerpt == 'summary' else '') + '.pkl')

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
def binarize_morality(models):
    for model in models:
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

    for c in config:
        if c == 1:
            excerpts = ['full_QnA']
            models = ['deepseek_bin_ao', 'chatgpt_bin_ao']
            compute_morality_source(models, excerpts)
        elif c == 2:
            compute_synthetic_data()
            compute_synthetic_morality()
        elif c == 3:
            models = ['lda', 'lda_resp', 'lda_sum']
            binarize_morality(models)