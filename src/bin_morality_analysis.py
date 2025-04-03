import os
import time

import numpy as np
import pandas as pd
import requests
import seaborn as sns
import spacy
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import minmax_scale, scale
from sklearn.utils import resample
from statsmodels.discrete.discrete_model import Probit
from statsmodels.regression.linear_model import OLS
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from transformers import pipeline
import patsy

from __init__ import *
from src.helpers import MORALITY_ORIGIN, MORALITY_ORIGIN_EXPLAINED, MORALITY_VOCAB, format_pvalue, llm_prompt
from src.parser import merge_summaries, prepare_data, wave_parser


#Compute morality dimensions from interviews
def compute_morality_dimensions(models, morality_texts):
    for morality_text in morality_texts:
        interviews = wave_parser()
        interviews = merge_summaries(interviews)

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

            data.to_pickle('data/cache/morality_model-' + model + ('_resp' if morality_text == 'Morality Response' else '_sum' if morality_text == 'Morality Summary' else '') + '.pkl')

            #Binarize continuous morality dimensions
            if model in ['sbert', 'lg', 'lda']:
                data[MORALITY_ORIGIN] = (data[MORALITY_ORIGIN].apply(minmax_scale) > .5).astype(int)
                data.to_pickle('data/cache/morality_model-' + model + ('_resp' if morality_text == 'Morality Response' else '_sum' if morality_text == 'Morality Summary' else '') + '_bin.pkl')        


#Plot F1 Score for all models
def evaluate_morality_dimensions(models, evaluation_waves, n_bootstraps, human_evaluation):
    #Prepare data
    interviews = prepare_data(models)
    interviews = pd.concat([pd.DataFrame(interviews[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN for model in models + ['gold', 'crowd']]].values, columns=[mo + '_' + model for mo in MORALITY_ORIGIN for model in models + ['gold', 'crowd']]) for wave in evaluation_waves]).dropna()
    print('Evaluation data size', len(interviews))

    #Bootstrapping
    scores = []
    for i in range(n_bootstraps):
        indices = resample(range(len(interviews)), replace=True, random_state=42 + i)
        data = interviews.iloc[indices]

        if human_evaluation:
            score = pd.DataFrame([{mo:f1_score(data[mo + '_gold'], data[mo + '_crowd'], average='weighted') for mo in MORALITY_ORIGIN}])
            score['Model'] = 'Crowdworkers'
            scores.append(round(score, 2))
        for model in models:
            score = pd.DataFrame([{mo:f1_score(data[mo + '_gold'], data[mo + '_' + model], average='weighted') for mo in MORALITY_ORIGIN}])
            score['Model'] = {'wc_bin':'$Dictionary$', 'wc_sum_bin':'$Dictionary_{Σ}$', 'wc_resp_bin':'$Dictionary_{R}$', 'lda_bin':'$LDA$', 'lda_sum_bin':'$LDA_{Σ}$', 'lda_resp_bin':'$LDA_{R}$', 'sbert_bin':'$SBERT$', 'sbert_sum_bin':'$SBERT_{Σ}$', 'sbert_resp_bin':'$SBERT_{R}$', 'nli_bin':'$NLI$', 'nli_sum_bin':'$NLI_{Σ}$', 'nli_resp_bin':'$NLI_{R}$', 'chatgpt_bin':'$GPT$', 'chatgpt_bin_3.5':'$GPT_{3.5}$', 'chatgpt_sum_bin':'$GPT_{Σ}$', 'chatgpt_resp_bin':'$GPT_{R}$', 'chatgpt_bin_nt':'$GPT_{NT}$', 'chatgpt_bin_ar':'$GPT_{AR}$', 'chatgpt_bin_toa':'$GPT_{TOA}$', 'chatgpt_bin_to1':'$GPT_{TO1}$', 'chatgpt_bin_rto1':'$GPT_{RTO1}$', 'chatgpt_bin_cto1':'$GPT_{CTO1}$', 'chatgpt_bin_dto1':'$GPT_{DTO1}$', 'deepseek_bin':'$DeepSeek$', 'deepseek_sum_bin':'$DeepSeek_{Σ}$', 'deepseek_resp_bin':'$DeepSeek_{R}$', 'deepseek_bin_nt':'$DeepSeek_{NT}$', 'deepseek_bin_ar':'$DeepSeek_{AR}$', 'deepseek_bin_toa':'$DeepSeek_{TOA}$', 'deepseek_bin_to1':'$DeepSeek_{TO1}$', 'deepseek_bin_rto1':'$DeepSeek_{RTO1}$', 'deepseek_bin_cto1':'$DeepSeek_{CTO1}$', 'deepseek_bin_dto1':'$DeepSeek_{DTO1}$'}.get(model, model)
            scores.append(round(score, 2))
    scores = pd.concat(scores, ignore_index=True).iloc[::-1]
    display(scores.set_index('Model').groupby('Model', sort=False).mean().round(2))
    scores.set_index('Model').groupby('Model', sort=False).mean().round(2).to_clipboard()
    scores['score'] = (scores[MORALITY_ORIGIN]).mean(axis=1).round(2)
    display(scores.set_index('Model').groupby('Model', sort=False).mean().round(2)[['score']])
    
    #Plot model comparison
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    sns.barplot(data=scores, y='Model', x='score')
    ax = plt.gca()
    ax.set_xlim(.4, .85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Weighted F1 Score')
    plt.ylabel('')
    plt.savefig('data/plots/fig-model_comparison.png', bbox_inches='tight')
    plt.show()

#Predict Survey and Oral Behavior based on Morality Origin
def predict_behavior(interviews, conf, to_latex):
    print(conf['Description'])

    #Run regressions with and without controls
    if conf['Controls']:
        simple_conf = conf.copy()
        simple_conf['Controls'] = []
        simple_conf['References'] = {'Attribute Names': [], 'Attribute Values': []}
        extended_confs = [simple_conf, conf]
    else:
        extended_confs = [conf]
    
    extended_results = []
    for conf in extended_confs:
        #Prepare Data
        data = interviews.copy()
        data[[wave + ':Wave' for wave in ['Wave 1', 'Wave 2', 'Wave 3']]] = pd.Series([wave.split()[1] for wave in ['Wave 1', 'Wave 2', 'Wave 3']])
        data = pd.concat([pd.DataFrame(data[['Survey Id'] + [from_wave + ':Wave'] + [from_wave + ':' + pr for pr in conf['Predictors']] + [from_wave + ':' + c for c in conf['Controls']] + ([from_wave + ':' + p for p in conf['Predictions']] if conf['Previous Behavior'] else []) + [to_wave + ':' + p for p in conf['Predictions']]].values) for from_wave, to_wave in zip(conf['From_Wave'], conf['To_Wave'])])
        data.columns = ['Survey Id'] + ['Wave'] + conf['Predictors'] + conf['Controls'] + (conf['Predictions'] if conf['Previous Behavior'] else []) + [p + '_pred' for p in conf['Predictions']]
        data = data.map(lambda x: np.nan if x == None else x)
        data = data[~data[conf['Predictors']].isna().all(axis=1)]
        
        #Binary Representation for Probit Model
        if conf['Model']  == 'Probit':
            data[(conf['Predictions'] if conf['Previous Behavior'] else []) + [p + '_pred' for p in conf['Predictions']]] = data[(conf['Predictions'] if conf['Previous Behavior'] else []) + [p + '_pred' for p in conf['Predictions']]].map(lambda p: int(p > .5) if not pd.isna(p) else pd.NA)

        #Add Reference Controls
        for attribute_name in conf['References']['Attribute Names']:
            dummies = pd.get_dummies(data[attribute_name], prefix=attribute_name, prefix_sep=' = ').astype(int)
            data = pd.concat([data, dummies], axis=1).drop(attribute_name, axis=1)
            c = 'Controls' if attribute_name in conf['Controls'] else 'Predictors' if attribute_name in conf['Predictors'] else None
            conf[c] = conf[c][:conf[c].index(attribute_name)] + list(dummies.columns) + conf[c][conf[c].index(attribute_name) + 1:]

        #Convert Data to Numeric
        data = data.apply(pd.to_numeric)

        #Compute Descriptive Statistics for Controls
        if conf['Controls']:
            stats = []
            for wave in conf['From_Wave']:
                slice = data[data['Wave'] == int(wave.split()[1])]
                stat = slice[conf['Controls']].describe(include = 'all').T[['count', 'mean', 'std', 'min', 'max']]
                stat[['count', 'mean', 'std', 'min', 'max']] = stat.apply(lambda s: pd.Series([s.iloc[0], round(s.iloc[1], 2), round(s.iloc[2], 2), s.iloc[3], s.iloc[4]]), axis=1).astype(pd.Series([int, float, float, int, int]))
                stats.append(stat)
            stats = pd.concat(stats, axis=1)
            stats.columns = pd.MultiIndex.from_tuples([(wave, stat) for wave in conf['From_Wave'] for stat in stats.columns[:5]])
            print(stats.to_latex()) if to_latex else display(stats)
            stats.to_clipboard()
        
        #Compute Descriptive Statistics for Predictions
        if conf['Predictions']:
            stats = []
            for wave in conf['From_Wave']:
                slice = data[data['Wave'] == int(wave.split()[1])]
                stat = slice[[p + '_pred' for p in conf['Predictions']]].describe(include = 'all').T[['count', 'mean', 'std', 'min', 'max']]
                stat = stat.map(lambda x: x if not pd.isna(x) else -1)
                stat[['count', 'mean', 'std', 'min', 'max']] = stat.apply(lambda s: pd.Series([s.iloc[0], round(s.iloc[1], 2), round(s.iloc[2], 2), s.iloc[3], s.iloc[4]]), axis=1).astype(pd.Series([int, float, float, int, int]))
                stat = stat.map(lambda x: x if not x == -1 else '-')
                stat['<NA>'] = slice[[p + '_pred' for p in conf['Predictions']]].isnull().sum()
                stats.append(stat)
            stats = pd.concat(stats, axis=1)
            stats.index = conf['Predictions']
            stats.columns = pd.MultiIndex.from_tuples([(wave, stat) for wave in conf['To_Wave'] for stat in stats.columns[:6]])
            print(stats.to_latex()) if to_latex else display(stats)
            stats.to_clipboard()
        
        #Compute Dummy for Wave variable
        if conf['Dummy']:
            dummies = pd.get_dummies(data['Wave'], prefix='Wave', prefix_sep=' = ').astype(int)
            dummies = dummies[dummies.columns[1:]]
            data = pd.concat([data, dummies], axis=1).drop('Wave', axis=1)

        #Drop NA and Reference Dummies
        conf['Controls'] = [c for c in conf['Controls'] if c not in [attribute_name + ' = ' + attribute_value for attribute_name, attribute_value in zip(conf['References']['Attribute Names'], conf['References']['Attribute Values'])]]
        conf['Predictors'] = [c for c in conf['Predictors'] if c not in [attribute_name + ' = ' + attribute_value for attribute_name, attribute_value in zip(conf['References']['Attribute Names'], conf['References']['Attribute Values'])]]
        data = data.drop([attribute_name + ' = ' + attribute_value for attribute_name, attribute_value in zip(conf['References']['Attribute Names'], conf['References']['Attribute Values'])], axis=1)
        data = data.reset_index(drop=True)

        #Compute Results
        if conf['Model'] in ['Probit', 'OLS']:
            #Define Formulas
            formulas = ['Q("' + p + '_pred")' + ' ~ ' + ' + '.join(['Q("' + pr + '")' for pr in conf['Predictors']]) + (' + ' + ' + '.join(['Q("' + c + '")' for c in conf['Controls']]) if conf['Controls'] else '') + ('+ Q("' + p + '")' if conf['Previous Behavior'] else '') + ' + Q("Survey Id")' + (' + ' + ' + '.join(['Q("' + w + '")' for w in dummies.columns]) if conf['Dummy'] and (p not in ['Cheat', 'Cutclass', 'Secret']) else '') + (' - 1' if not conf['Intercept'] else '') for p in conf['Predictions']]
            
            #Run Regressions
            results = {}
            results_index = (['Intercept'] if conf['Intercept'] else []) + [pr.split('_')[0] for pr in conf['Predictors']] + conf['Controls'] + ([w for w in dummies.columns] if conf['Dummy'] else []) + (['Previous Behavior'] if conf['Previous Behavior'] else []) + ['N', 'AIC']
            for formula, p in zip(formulas, conf['Predictions']):
                y, X = patsy.dmatrices(formula, data, return_type='dataframe')
                groups = X['Q("Survey Id")']
                X = X.drop('Q("Survey Id")', axis=1)
                model = Probit if conf['Model'] == 'Probit' else OLS if conf['Model'] == 'OLS' else None
                fit_params = {'method':'bfgs', 'disp':False} if conf['Model'] == 'Probit' else {'cov':'cluster', 'cov_kwds':{'groups': groups}} if conf['Model'] == 'OLS' else {}
                model = model(y, X).fit(maxiter=10000, **fit_params)
                result = {param:(coef,pvalue) for param, coef, pvalue in zip(model.params.index, model.params, model.pvalues)}
                if conf['Previous Behavior']:
                    result['Previous Behavior'] = result['Q("' + p + '")']
                    result.pop('Q("' + p + '")')
                result['N'] = int(model.nobs)
                result['AIC'] = round(model.aic, 2)
                results[p.split('_')[0]] = result
            results = pd.DataFrame(results)
            results.index = results_index

            #Scale Results
            results = pd.concat([pd.DataFrame(('(' + pd.DataFrame(scale(results[:-2].map(lambda c: c[0] if not pd.isna(c) else None))).map(str) + ',' + pd.DataFrame(results[:-2].map(lambda c: c[1] if not pd.isna(c) else None)).map(str).values + ')').values, index=results[:-2].index, columns=results[:-2].columns).map(str).replace('(nan,nan)', 'None').map(eval).map(format_pvalue), pd.DataFrame(results[-2:])])
        
        #Compute Results
        elif conf['Model'] in ['Pearson']:
            #Compute Correlations
            results = pd.DataFrame(index=[mo1 + ' - ' + mo2 for i, mo1 in enumerate(MORALITY_ORIGIN) for j, mo2 in enumerate(MORALITY_ORIGIN) if i < j] + ['N'], columns=list(set([c.split('_')[1] for c in conf['Predictors']])))
            for estimator in list(set([c.split('_')[1] for c in conf['Predictors']])):
                slice = data[[mo + '_' + estimator + '_bin' for mo in MORALITY_ORIGIN]].dropna().reset_index(drop=True)
                for i in results.index[:-1]:
                    results.loc[i, estimator] = format_pvalue(pearsonr(slice[i.split(' - ')[0] + '_' + estimator + '_bin'], slice[i.split(' - ')[1] + '_' + estimator + '_bin']))
                results.loc['N', estimator] = len(slice)

        extended_results.append(results)
    
    #Concatenate Results with and without controls
    results = pd.concat(extended_results, axis=1).fillna('-')
    results = results[[pr.split('_')[0] for pr in conf['Predictions']] if conf['Predictions'] else results.columns]
    if conf['Model'] == 'Probit':
        results = pd.concat([results.drop(index=['N', 'AIC']), results.loc[['N', 'AIC']]])
    print(results.to_latex()) if to_latex else display(results)
    results.to_clipboard()


if __name__ == '__main__':
    #Hyperparameters
    config = [2]

    for c in config:

        if c == 1:
            morality_texts = ['Morality Text']
            models = ['deepseek_bin', 'chatgpt_bin']
            compute_morality_dimensions(models, morality_texts)

        elif c == 2:
            #Prompt engineering
            # models = ['deepseek_bin', 'deepseek_bin_dto1', 'deepseek_bin_cto1', 'deepseek_bin_rto1', 'deepseek_bin_to1', 'deepseek_bin_toa', 'chatgpt_bin', 'chatgpt_bin_dto1', 'chatgpt_bin_cto1', 'chatgpt_bin_rto1', 'chatgpt_bin_to1', 'chatgpt_bin_toa']
            #Speaker distinction
            # models = ['deepseek_bin', 'deepseek_bin_ar', 'deepseek_bin_nt', 'chatgpt_bin', 'chatgpt_bin_ar', 'chatgpt_bin_nt']
            #Input engineering
            # models = ['nli_sum_bin', 'nli_resp_bin', 'nli_bin', 'sbert_sum_bin', 'sbert_resp_bin', 'sbert_bin', 'lda_sum_bin', 'lda_resp_bin', 'lda_bin', 'wc_sum_bin', 'wc_resp_bin', 'wc_bin']
            #Final model comparison
            models = ['deepseek_bin', 'chatgpt_bin', 'nli_sum_bin', 'sbert_resp_bin', 'lda_sum_bin', 'wc_bin']
            evaluation_waves = ['Wave 1']
            human_evaluation = False
            n_bootstraps = 10
            evaluate_morality_dimensions(models=models, evaluation_waves=evaluation_waves, n_bootstraps=n_bootstraps, human_evaluation=human_evaluation)

        elif c == 3:
            to_latex = False
            model = 'deepseek_bin'
            interviews = prepare_data([model])
            #Future Behavior with Controls
            conf = {'Description': 'Predicting Future Behavior: ' + model,
                    'From_Wave': ['Wave 1', 'Wave 2', 'Wave 3'],
                    'To_Wave': ['Wave 2', 'Wave 3', 'Wave 4'],
                    'Predictors': [mo + '_' + model for mo in MORALITY_ORIGIN],
                    'Predictions': ['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help'],
                    'Dummy' : True,
                    'Intercept': True,
                    'Previous Behavior': True,
                    'Model': 'Probit',
                    'Controls': ['Number of friends', 'Regular volunteers', 'Use drugs', 'Similar beliefs', 'Religion', 'Race', 'Gender', 'Region', 'Parent Education', 'Household Income', 'GPA', 'Age'],
                    'References': {'Attribute Names': ['Religion', 'Race', 'Gender', 'Region', 'Parent Education'], 'Attribute Values': ['Catholic', 'White', 'Male', 'Not South', '≥ College']}}
            #Computing Pairwise Correlations
            # conf = {'Description': 'Computing Pairwise Correlations: ' + model, 'From_Wave': ['Wave 1', 'Wave 2', 'Wave 3'], 'To_Wave': ['Wave 1', 'Wave 2', 'Wave 3'], 'Predictors': [mo + '_' + model for mo in MORALITY_ORIGIN], 'Predictions': [], 'Previous Behavior': False, 'Model': 'Pearson', 'Controls': [], 'References': {'Attribute Names': [], 'Attribute Values': []}}
            #Estimating Morality from Social Categories (OLS)
            # conf = {'Description': 'Estimating Morality Sources from Social Categories (OLS): ' + model, 'From_Wave': ['Wave 1', 'Wave 2', 'Wave 3'], 'To_Wave': ['Wave 1', 'Wave 2', 'Wave 3'], 'Predictors': ['Verbosity', 'Uncertainty', 'Complexity', 'Sentiment'], 'Predictions': [mo + '_' + model for mo in MORALITY_ORIGIN], 'Dummy' : True, 'Intercept': True, 'Previous Behavior': False, 'Model': 'OLS', 'Controls': ['Number of friends', 'Regular volunteers', 'Use drugs', 'Similar beliefs', 'Religion', 'Race', 'Gender', 'Region', 'Parent Education', 'Household Income', 'GPA'], 'References': {'Attribute Names': ['Religion', 'Race', 'Gender', 'Region', 'Parent Education'], 'Attribute Values': ['Catholic', 'White', 'Male', 'Not South', '≥ College']}}
            predict_behavior(interviews, conf, to_latex)