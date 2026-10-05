import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import minmax_scale, scale
from sklearn.utils import resample
from statsmodels.discrete.discrete_model import Probit
from statsmodels.regression.linear_model import OLS
import patsy

from __init__ import *
from src.helpers import CODERS, DEVELOPMENT_WAVES, LLM_REGISTRY, MORALITY_ORIGIN, MORALITY_ORIGIN_EXPLAINED, MORALITY_VOCAB, TEST_WAVES, format_pvalue, llm_prompt
from src.parser import add_crowd_labels, load_text_dump, merge_summaries, prepare_crowd_task, prepare_data, update_cache, wave_parser

#Heavy NLP dependencies (torch, transformers, spaCy, gensim, sentence-transformers) are imported lazily inside
#compute_morality_dimensions, so that the evaluation and regression analyses run without them.

INPUT_SUFFIX = {'Morality Text': '', 'Morality Response': '_resp', 'Morality Summary': '_sum'}


#Name of a model's binary output in the data dump, e.g. ('deepseek_bin', 'Morality Response') -> 'deepseek_resp_bin'
def cache_name(model, morality_text):
    base = model[:-len('_bin')] if model.endswith('_bin') else model.replace('_quant', '')
    return base + INPUT_SUFFIX[morality_text] + '_bin'


#Seeded (guided) LDA: one topic per moral dimension, seeded with that dimension's vocabulary through an
#asymmetric topic-word prior; fitted on the development wave(s) only and then applied, frozen, to all waves
def seeded_lda(data, morality_text, fit_mask, nlp_model, seed_prior=1., base_prior=.01, passes=50):
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    keep = lambda w: w.is_alpha and not w.is_stop and w.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']
    docs = data[morality_text].fillna('').apply(lambda t: [w.lemma_.lower() for w in nlp_model(t) if keep(w)]).tolist()
    fit_docs = [d for d, m in zip(docs, fit_mask) if m]

    dictionary = Dictionary(fit_docs)
    seeds = {mo: [w for w in MORALITY_VOCAB[mo] if w in dictionary.token2id] for mo in MORALITY_ORIGIN}
    dictionary.filter_extremes(no_below=2, no_above=.9, keep_tokens=[w for ws in seeds.values() for w in ws])
    print('Seed words found in the development texts:', seeds)

    eta = np.full((len(MORALITY_ORIGIN), len(dictionary)), base_prior)
    for k, mo in enumerate(MORALITY_ORIGIN):
        for w in seeds[mo]:
            eta[:, dictionary.token2id[w]] = base_prior / 100
            eta[k, dictionary.token2id[w]] = seed_prior

    lda = LdaModel([dictionary.doc2bow(d) for d in fit_docs], id2word=dictionary, num_topics=len(MORALITY_ORIGIN), eta=eta, alpha='symmetric', passes=passes, iterations=400, random_state=42)
    theta = [dict(lda.get_document_topics(dictionary.doc2bow(d), minimum_probability=0)) for d in docs]
    return pd.DataFrame([[t.get(k, 0.) for k in range(len(MORALITY_ORIGIN))] for t in theta], columns=MORALITY_ORIGIN, index=data.index)


#Call an LLM of LLM_REGISTRY (OpenAI-compatible API) on one transcript; returns (label, record for the audit log)
def call_llm(llm, prompt, text, timeout=60, max_retries=8, backoff_factor=1.):
    cfg = LLM_REGISTRY[llm]
    headers = {'Content-Type': 'application/json'} | ({'Authorization': 'Bearer ' + os.getenv(cfg['key'], '')} if cfg['key'] else {})
    data = {'model': cfg['model'], 'messages': [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': text}], cfg.get('token_field', 'max_tokens'): cfg['max_tokens'], 'seed': 42}
    data |= ({'temperature': cfg['temperature']} if cfg['temperature'] is not None else {}) | cfg.get('extra', {})

    for attempt in range(max_retries):
        try:
            response = requests.post(cfg['url'] + '/chat/completions', json=data, headers=headers, timeout=timeout)
            if response.status_code == 429:
                time.sleep(float(response.headers.get('Retry-After', backoff_factor * (2 ** attempt))))
                continue
            response.raise_for_status()
            response = response.json()
            content = response['choices'][0]['message'].get('content') or ''
            answer = re.sub(r'<think>.*?</think>', '', content, flags=re.S)
            label = 0 if '0' in answer else 1 if '1' in answer else -1
            if label == -1:
                raise Exception('Response not parsable: ' + content)
            return label, {'returned_model': response.get('model'), 'system_fingerprint': response.get('system_fingerprint'), 'usage': response.get('usage'), 'response': content}
        except Exception as e:
            print(f'Attempt {attempt + 1} failed: {str(e)}')
            time.sleep(backoff_factor * (2 ** attempt))

    print('Request failed after max retries')
    return -1, {'error': 'failed after max retries'}


#Annotate the transcripts with an LLM, one call per (interview, moral dimension) with the prompt of the paper.
#Every call is appended to an audit log (exact model returned by the provider, usage, timestamp); labels already
#in the log are reused, so an interrupted run resumes where it stopped. Returns one 0/1 column per dimension.
def annotate_with_llm(data, llm, morality_text, log_dir='data/cache/llm_logs', workers=4):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, cache_name(llm, morality_text) + '.jsonl')
    labels = {}
    if os.path.isfile(log_file):
        for line in open(log_file):
            record = json.loads(line)
            if record['label'] != -1:
                labels[(record['Interview Code'], record['Wave'], record['Dimension'])] = record['label']

    jobs = [(row['Interview Code'], int(row['Wave']), mo, row[morality_text]) for _, row in data.iterrows() for mo in MORALITY_ORIGIN if not pd.isna(row[morality_text]) and (row['Interview Code'], int(row['Wave']), mo) not in labels]
    print(llm, '| model:', LLM_REGISTRY[llm]['model'], '|', len(labels), 'labels in the log,', len(jobs), 'calls to make')

    def work(job):
        code, wave, mo, text = job
        label, record = call_llm(llm, llm_prompt(mo, 'bin'), text)
        return record | {'Interview Code': code, 'Wave': wave, 'Dimension': mo, 'label': label, 'model': LLM_REGISTRY[llm]['model'], 'timestamp': datetime.now(timezone.utc).isoformat()}

    with open(log_file, 'a') as log, ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(work, jobs):
            log.write(json.dumps(record) + '\n')
            labels[(record['Interview Code'], record['Wave'], record['Dimension'])] = record['label']

    return pd.DataFrame({mo: [labels.get((row['Interview Code'], int(row['Wave']), mo), -1) if not pd.isna(row[morality_text]) else pd.NA for _, row in data.iterrows()] for mo in MORALITY_ORIGIN}, index=data.index)


#Compute morality dimensions from interviews
#interviews: DataFrame with 'Interview Code', 'Wave' and the text columns (default: parse the transcripts)
#fit_waves : waves on which the guided LDA and its score scaling are estimated (the development wave)
def compute_morality_dimensions(models, morality_texts, interviews=None, fit_waves=(1,)):
    if interviews is None:
        interviews = merge_summaries(wave_parser())
    for morality_text in morality_texts:

        #Compute all models
        for model in models:
            data = interviews.copy()
            fit_mask = data['Wave'].isin(fit_waves).values
            fit_mask = fit_mask if fit_mask.any() else np.ones(len(data), dtype=bool)

            #NLI model
            if model in ['nli_quant', 'nli_quant_uv']:
                import torch
                from transformers import pipeline
                #Premise and hypothesis templates
                hypothesis_template = 'The reasoning in this example is based on {}.'
                model_params = {'device':0} if torch.cuda.is_available() else {}
                morality_pipeline = pipeline('zero-shot-classification', model='roberta-large-mnli', **model_params)

                #Trasformation functions
                classifier = lambda series: pd.Series(morality_pipeline(series.tolist(), list(MORALITY_ORIGIN_EXPLAINED.keys()), hypothesis_template=hypothesis_template, multi_label=(True if model == 'nli_quant' else False if model == 'nli_quant_uv' else None)))
                aggregator = lambda r: pd.DataFrame([{MORALITY_ORIGIN_EXPLAINED[l]:s for l, s in zip(r['labels'], r['scores'])}]).max()

                #Classify morality origin and join results
                morality_origin = classifier(data[morality_text]).apply(aggregator)
                data = data.join(morality_origin)

            #LLM models (see LLM_REGISTRY in helpers)
            elif model in LLM_REGISTRY:
                data[MORALITY_ORIGIN] = annotate_with_llm(data, model, morality_text)

            #SBERT model
            elif model == 'sbert':
                import torch
                from sentence_transformers import SentenceTransformer
                from torch.nn.functional import cosine_similarity
                #Compute embeddings
                nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
                vectors = pd.DataFrame(nlp_model.encode(data[morality_text])).apply(np.array, axis=1)

                #Compute cosine similarity with morality origin vectors
                morality_origin = pd.Series({mo:nlp_model.encode(mo) for mo in MORALITY_ORIGIN})
                data[MORALITY_ORIGIN] = pd.DataFrame([vectors.apply(lambda e: cosine_similarity(torch.from_numpy(e).view(1, -1), torch.from_numpy(morality_origin[mo]).view(1, -1)).numpy()[0]) for mo in MORALITY_ORIGIN], index=MORALITY_ORIGIN).T

            #SpaCy model
            elif model == 'lg':
                import spacy
                import torch
                from torch.nn.functional import cosine_similarity
                #Compute embeddings
                nlp_model = spacy.load('en_core_web_lg')
                vectors = data[morality_text].apply(lambda s: np.mean([w.vector for w in nlp_model(s) if w.pos_ in ['NOUN', 'ADJ', 'VERB']], axis=0) if not pd.isna(s) else s)

                #Compute cosine similarity with morality origin vectors
                morality_origin = pd.Series({mo:nlp_model(mo).vector for mo in MORALITY_ORIGIN})
                data[MORALITY_ORIGIN] = pd.DataFrame([vectors.apply(lambda e: cosine_similarity(torch.from_numpy(e).view(1, -1), torch.from_numpy(morality_origin[mo]).view(1, -1)).numpy()[0]) for mo in MORALITY_ORIGIN], index=MORALITY_ORIGIN).T

            #Guided LDA model (seeded with MORALITY_VOCAB)
            elif model == 'lda':
                import spacy
                nlp_model = spacy.load('en_core_web_lg')
                data[MORALITY_ORIGIN] = seeded_lda(data, morality_text, fit_mask, nlp_model)

            #Dictionary model (MORALITY_VOCAB)
            elif model == 'wc':
                import spacy
                nlp_model = spacy.load('en_core_web_lg')
                lemmas = data[morality_text].fillna('').apply(lambda t: [w.lemma_.lower() for w in nlp_model(t)])
                data[MORALITY_ORIGIN] = pd.DataFrame({mo: lemmas.apply(lambda l: int(any(w in MORALITY_VOCAB[mo] for w in l))) for mo in MORALITY_ORIGIN})

            if model not in LLM_REGISTRY:
                data.to_pickle('data/cache/morality_model-' + model + INPUT_SUFFIX[morality_text] + '.pkl')

            #Binarize continuous morality dimensions (guided LDA: scaling estimated on the development wave)
            if model in ['nli_quant', 'sbert', 'lg']:
                data[MORALITY_ORIGIN] = (data[MORALITY_ORIGIN].apply(minmax_scale) > .5).astype(int)
            elif model == 'lda':
                lo, hi = data.loc[fit_mask, MORALITY_ORIGIN].min(), data.loc[fit_mask, MORALITY_ORIGIN].max()
                data[MORALITY_ORIGIN] = (((data[MORALITY_ORIGIN] - lo) / (hi - lo)) > .5).astype(int)
            data.to_pickle('data/cache/morality_model-' + cache_name(model, morality_text) + '.pkl')


#Write binary model outputs (pickles that contain 'Survey Id') into the data dump data/cache/morality.csv
def export_to_cache(names):
    columns = []
    for name in names:
        data = pd.read_pickle('data/cache/morality_model-' + name + '.pkl')
        data[MORALITY_ORIGIN] = data[MORALITY_ORIGIN].apply(pd.to_numeric, errors='coerce').replace(-1, np.nan)
        wide = data.pivot_table(index='Survey Id', columns='Wave', values=MORALITY_ORIGIN)
        wide.columns = ['Wave ' + str(w) + ':' + mo + '_' + name for mo, w in wide.columns]
        columns.append(wide)
    update_cache(pd.concat(columns, axis=1).reset_index())


#Annotate interviews (waves of the transcripts, or of the local text dump) with one model and morality text
def annotate(model, waves, morality_texts, source='transcripts', gold=None):
    if source == 'transcripts':
        interviews = wave_parser(wave_filter=waves)
        interviews = merge_summaries(interviews) if 'Morality Summary' in morality_texts else interviews
    else:
        interviews = load_text_dump()
        interviews = interviews[interviews['Wave'].isin(waves)].reset_index(drop=True)
    compute_morality_dimensions([model], morality_texts, interviews, fit_waves=[int(w.split()[1]) for w in DEVELOPMENT_WAVES])
    names = [cache_name(model, t) for t in morality_texts]
    print('Saved', ', '.join('data/cache/morality_model-' + n + '.pkl' for n in names))

    #Waves 1-3 of the text dump are keyed by Survey Id and can be written into the data dump
    if source == 'dump' and set(waves) <= {1, 2, 3}:
        export_to_cache(names)

    #Optional evaluation against gold labels (CSV with 'Interview Code', 'Wave' and one 0/1 column per dimension)
    if gold:
        gold = pd.read_csv(gold)
        for name in names:
            data = pd.read_pickle('data/cache/morality_model-' + name + '.pkl').merge(gold, on=['Interview Code', 'Wave'], suffixes=('', '_gold')).dropna()
            scores = {mo: f1_score(data[mo + '_gold'].astype(int), data[mo].astype(int), average='weighted') for mo in MORALITY_ORIGIN}
            print(name, '| N =', len(data), '| weighted F1:', {mo: round(s, 2) for mo, s in scores.items()}, '| mean:', round(np.mean(list(scores.values())), 2))


MODEL_LABELS = {'crowd': 'Crowdworkers', 'Coder_1': 'Coder 1', 'Coder_2': 'Coder 2',
                'wc_bin': '$Dictionary$', 'wc_sum_bin': '$Dictionary_{Σ}$', 'wc_resp_bin': '$Dictionary_{R}$',
                'lda_bin': '$LDA$', 'lda_sum_bin': '$LDA_{Σ}$', 'lda_resp_bin': '$LDA_{R}$',
                'sbert_bin': '$SBERT$', 'sbert_sum_bin': '$SBERT_{Σ}$', 'sbert_resp_bin': '$SBERT_{R}$',
                'nli_bin': '$NLI$', 'nli_sum_bin': '$NLI_{Σ}$', 'nli_resp_bin': '$NLI_{R}$',
                'chatgpt_bin': '$GPT$', 'chatgpt_bin_3.5': '$GPT_{3.5}$', 'chatgpt_sum_bin': '$GPT_{Σ}$', 'chatgpt_resp_bin': '$GPT_{R}$', 'chatgpt_bin_nt': '$GPT_{NT}$', 'chatgpt_bin_ar': '$GPT_{AR}$', 'chatgpt_bin_toa': '$GPT_{TOA}$', 'chatgpt_bin_to1': '$GPT_{TO1}$', 'chatgpt_bin_rto1': '$GPT_{RTO1}$', 'chatgpt_bin_cto1': '$GPT_{CTO1}$', 'chatgpt_bin_dto1': '$GPT_{DTO1}$',
                'deepseek_bin': '$DeepSeek$', 'deepseek_sum_bin': '$DeepSeek_{Σ}$', 'deepseek_resp_bin': '$DeepSeek_{R}$', 'deepseek_bin_nt': '$DeepSeek_{NT}$', 'deepseek_bin_ar': '$DeepSeek_{AR}$', 'deepseek_bin_toa': '$DeepSeek_{TOA}$', 'deepseek_bin_to1': '$DeepSeek_{TO1}$', 'deepseek_bin_rto1': '$DeepSeek_{RTO1}$', 'deepseek_bin_cto1': '$DeepSeek_{CTO1}$', 'deepseek_bin_dto1': '$DeepSeek_{DTO1}$'}


plain_label = lambda model: re.sub(r'[$}]', '', MODEL_LABELS.get(model, model)).replace('_{', ' ')


#Weighted F1 of each model against the gold standard (or another reference annotator) on the given waves, with
#95% percentile bootstrap intervals of the average over the moral dimensions; optionally plotted (models are
#drawn in the given order from bottom to top, as in the paper)
def evaluate_morality_dimensions(interviews, models, evaluation_waves, n_bootstraps=1000, plot_file=None, reference='gold'):
    missing = [m for m in models if not all(interviews.get(wave + ':' + mo + '_' + m, pd.Series(dtype=float)).notna().any() for wave in evaluation_waves for mo in MORALITY_ORIGIN)]
    if missing:
        print('No annotations for', evaluation_waves, '(skipped):', missing)
    models = [m for m in models if m not in missing]
    data = pd.concat([pd.DataFrame(interviews[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN for model in models + [reference]]].values, columns=[mo + '_' + model for mo in MORALITY_ORIGIN for model in models + [reference]]) for wave in evaluation_waves]).dropna().astype(int)
    print('Evaluation waves:', evaluation_waves, '| N =', len(data))

    score = lambda slice, model: [f1_score(slice[mo + '_' + reference], slice[mo + '_' + model], average='weighted') for mo in MORALITY_ORIGIN]
    results = pd.DataFrame([score(data, model) for model in models], columns=MORALITY_ORIGIN, index=models)
    results['Overall'] = results.mean(axis=1)
    if n_bootstraps:
        boots = pd.DataFrame([[np.mean(score(slice, model)) for model in models] for slice in (data.iloc[resample(range(len(data)), replace=True, random_state=42 + i)] for i in range(n_bootstraps))], columns=models)
        results['CI low'], results['CI high'] = boots.quantile(.025), boots.quantile(.975)
    results['N'] = len(data)
    results.index = [MODEL_LABELS.get(m, m) for m in models]
    display(results.round(2))

    if plot_file:
        sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
        plt.figure(figsize=(10, max(4, .75 * len(models) + 1)))
        ax = plt.gca()
        ax.barh(results.index, results['Overall'], color=sns.color_palette()[0], xerr=[results['Overall'] - results['CI low'], results['CI high'] - results['Overall']], error_kw={'ecolor': '.25', 'elinewidth': 1.5})
        ax.set_xlim(min(.4, np.floor(results['CI low'].min() * 20) / 20), max(.85, np.ceil(results['CI high'].max() * 20) / 20))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Weighted F1 Score')
        plt.ylabel('')
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        print('Saved', plot_file)
    return results


#Candidate configurations of each method (input variants; prompt variants for the LLMs)
METHOD_CONFIGURATIONS = {'Dictionary': ['wc_bin', 'wc_resp_bin', 'wc_sum_bin'],
                         'LDA': ['lda_bin', 'lda_resp_bin', 'lda_sum_bin'],
                         'SBERT': ['sbert_bin', 'sbert_resp_bin', 'sbert_sum_bin'],
                         'NLI': ['nli_bin', 'nli_resp_bin', 'nli_sum_bin'],
                         'GPT': ['chatgpt_bin', 'chatgpt_bin_dto1', 'chatgpt_bin_cto1', 'chatgpt_bin_rto1', 'chatgpt_bin_to1', 'chatgpt_bin_toa', 'chatgpt_resp_bin', 'chatgpt_sum_bin'],
                         'DeepSeek': ['deepseek_bin', 'deepseek_bin_dto1', 'deepseek_bin_cto1', 'deepseek_bin_rto1', 'deepseek_bin_to1', 'deepseek_bin_toa', 'deepseek_resp_bin', 'deepseek_sum_bin']}

#Select the configuration of each method with the highest weighted F1 on the development wave; it is then frozen
def select_configurations(interviews):
    scores = evaluate_morality_dimensions(interviews, [m for ms in METHOD_CONFIGURATIONS.values() for m in ms], DEVELOPMENT_WAVES, n_bootstraps=0)['Overall']
    selected = {method: max(ms, key=lambda m: scores[MODEL_LABELS[m]]) for method, ms in METHOD_CONFIGURATIONS.items()}
    print('Configurations selected on', DEVELOPMENT_WAVES, ':', selected)
    return selected


#File name of a figure/table; evaluations on non-default waves get the waves as suffix
output_file = lambda prefix, waves, extension: prefix + ''.join('_' + w.lower().replace(' ', '_') for w in (waves or [])) + extension

#Figures of the paper: 1-2 on the development wave, 3-4 on the held-out test wave (waves can be overridden)
def figure(number, interviews, waves=None):
    file = output_file('data/plots/figure_' + str(number), waves, '.png')
    if number in [1, 2]:
        waves = waves or DEVELOPMENT_WAVES
        models = ['nli_sum_bin', 'nli_resp_bin', 'nli_bin', 'sbert_sum_bin', 'sbert_resp_bin', 'sbert_bin', 'lda_sum_bin', 'lda_resp_bin', 'lda_bin', 'wc_sum_bin', 'wc_resp_bin', 'wc_bin'] if number == 1 else ['deepseek_bin', 'deepseek_bin_dto1', 'deepseek_bin_cto1', 'deepseek_bin_rto1', 'deepseek_bin_to1', 'deepseek_bin_toa', 'chatgpt_bin', 'chatgpt_bin_dto1', 'chatgpt_bin_cto1', 'chatgpt_bin_rto1', 'chatgpt_bin_to1', 'chatgpt_bin_toa']
    elif number == 3:
        waves = waves or TEST_WAVES
        selected = select_configurations(interviews)
        models = ['crowd'] + [selected[method] for method in ['DeepSeek', 'GPT', 'NLI', 'SBERT', 'LDA', 'Dictionary']]
    elif number == 4:
        waves = waves or TEST_WAVES
        models = ['deepseek_bin', 'deepseek_bin_ar', 'deepseek_bin_nt', 'chatgpt_bin', 'chatgpt_bin_ar', 'chatgpt_bin_nt']
    return evaluate_morality_dimensions(interviews, models, waves, plot_file=file)


#Agreement between the two trained coders and share of codes adjudicated by the postdoc (per wave and dimension)
def coder_reliability(interviews, waves=['Wave 1', 'Wave 3']):
    rows = []
    for wave in waves:
        data = interviews[[wave + ':' + mo + '_' + a for mo in MORALITY_ORIGIN for a in CODERS + ['gold']]].dropna().astype(int)
        for mo in MORALITY_ORIGIN:
            coder_1, coder_2, gold = (data[wave + ':' + mo + '_' + a] for a in CODERS + ['gold'])
            rows.append({'Wave': wave, 'Dimension': mo, 'N': len(data), 'Agreement': (coder_1 == coder_2).mean(), 'Kappa': cohen_kappa_score(coder_1, coder_2), 'Adjudicated': (coder_1 != coder_2).mean(), 'Adjudicated as present': gold[coder_1 != coder_2].mean()})
    return pd.DataFrame(rows).round(2)


#Agreement of each model with each individual trained coder (the other coder is included as human benchmark)
def compare_with_coders(interviews, models, waves=['Wave 1', 'Wave 3']):
    rows = []
    for wave in waves:
        for reference, other in [(CODERS[1], CODERS[0]), (CODERS[0], CODERS[1])]:
            for model in [other] + models:
                if interviews.get(wave + ':Intuitive_' + model, pd.Series(dtype=float)).notna().any():
                    data = interviews[[wave + ':' + mo + '_' + a for mo in MORALITY_ORIGIN for a in [reference, model]]].dropna().astype(int)
                    rows.append({'Wave': wave, 'Reference': reference, 'Annotator': model, 'N': len(data),
                                 'Weighted F1': np.mean([f1_score(data[wave + ':' + mo + '_' + reference], data[wave + ':' + mo + '_' + model], average='weighted') for mo in MORALITY_ORIGIN]),
                                 'Kappa': np.mean([cohen_kappa_score(data[wave + ':' + mo + '_' + reference], data[wave + ':' + mo + '_' + model]) for mo in MORALITY_ORIGIN])})
    rows = pd.DataFrame(rows)
    rows['Annotator'] = rows['Annotator'].map(lambda m: 'Other coder' if m in CODERS else plain_label(m))
    return rows.groupby(['Wave', 'Annotator'], sort=False)[['Weighted F1', 'Kappa']].mean().round(2)


#Interviews misclassified by the dictionary (with the predictions of all frozen models and the excerpt), as
#candidates for the qualitative analysis: error='fp' (false positives) or 'fn' (false negatives)
def misclassifications(interviews, dimension, error, waves):
    import spacy
    nlp_model = spacy.load('en_core_web_lg')
    texts = load_text_dump()
    #The dictionary on full transcripts (its errors are the point of the analysis); the other methods as frozen
    selected = select_configurations(interviews) | {'Dictionary': 'wc_bin'}
    methods = ['Dictionary', 'LDA', 'SBERT', 'NLI', 'DeepSeek', 'GPT']
    rows = []
    for wave in waves:
        data = interviews[['Survey Id', wave + ':' + dimension + '_gold'] + [wave + ':' + dimension + '_' + selected[m] for m in methods]].dropna()
        data.columns = ['Survey Id', 'Gold'] + methods
        data = data[(data['Gold'] == (0 if error == 'fp' else 1)) & (data['Dictionary'] == (1 if error == 'fp' else 0))]
        data = data.merge(texts[texts['Wave'] == int(wave.split()[1])][['Survey Id', 'Wave', 'Morality Text']], on='Survey Id')
        rows.append(data)
    data = pd.concat(rows, ignore_index=True)
    #For false positives, flag excerpts in which the dictionary terms occur only in the interviewer's questions
    if error == 'fp':
        terms_in = lambda text, speaker: any(w.lemma_.lower() in MORALITY_VOCAB[dimension] for line in text.split('\n') if line.startswith(speaker) for w in nlp_model(line[2:]))
        data['Terms only in questions'] = data['Morality Text'].apply(lambda t: terms_in(t, 'I:') and not terms_in(t, 'R:'))
    return data


#Specifications of the regressions (Table 5 and Appendix Tables A2-A9)
CONTROL_SETS = {'network': {'Controls': ['Number of friends', 'Regular volunteers', 'Use drugs', 'Similar beliefs'], 'References': {}},
                'religion': {'Controls': ['Religion'], 'References': {'Religion': 'Catholic'}},
                'demographics': {'Controls': ['Race', 'Gender', 'Region', 'Parent Education', 'Household Income', 'GPA', 'Age'], 'References': {'Race': 'White', 'Gender': 'Male', 'Region': 'Not South', 'Parent Education': '≥ College'}}}
CONTROL_SETS['all'] = {'Controls': [c for s in CONTROL_SETS.values() for c in s['Controls']], 'References': {k: v for s in CONTROL_SETS.values() for k, v in s['References'].items()}}

def regression_conf(model, control_set):
    return {'Description': 'Predicting Future Behavior: ' + model + ' | controls: ' + control_set,
            'From_Wave': ['Wave 1', 'Wave 2', 'Wave 3'],
            'To_Wave': ['Wave 2', 'Wave 3', 'Wave 4'],
            'Predictors': [mo + '_' + model for mo in MORALITY_ORIGIN],
            'Predictions': ['Pot', 'Drink', 'Volunteer', 'Help'],
            'Dummy' : True,
            'Intercept': True,
            'Previous Behavior': True,
            'Model': 'Probit',
            'Controls': list(CONTROL_SETS[control_set]['Controls']),
            'References': {'Attribute Names': list(CONTROL_SETS[control_set]['References']), 'Attribute Values': list(CONTROL_SETS[control_set]['References'].values())}}

#Regression table: for each outcome, the model without controls followed by one model per control set
def regression_table(interviews, model, control_sets):
    columns = {}
    for i, control_set in enumerate(control_sets):
        results, descriptives = predict_behavior(interviews, regression_conf(model, control_set), to_latex=False)
        for outcome in regression_conf(model, control_set)['Predictions']:
            columns.setdefault((outcome, '(1) no controls'), results[outcome].iloc[:, 0])
            columns[(outcome, '(' + str(i + 2) + ') ' + control_set)] = results[outcome].iloc[:, 1]
    columns = {k: columns[k] for k in sorted(columns, key=lambda k: regression_conf(model, 'all')['Predictions'].index(k[0]))}
    table = pd.concat(columns, axis=1)
    rows = list(dict.fromkeys([r for c in columns.values() for r in c.index if r not in ['N', 'AIC']])) + ['N', 'AIC']
    return table.reindex(rows).fillna('-'), descriptives


#Tables of the paper (and of the revision); tables with interview excerpts (3, 4) need the local text dump
def table(name, interviews, waves=None):
    os.makedirs('data/tables', exist_ok=True)
    file = output_file('data/tables/table_' + name, waves, '.csv')
    if name == 'A1':
        result = pd.DataFrame({mo: [', '.join(MORALITY_VOCAB[mo])] for mo in MORALITY_ORIGIN}, index=['Vocabulary']).T
    elif name == '2':
        selected = select_configurations(interviews)
        result = evaluate_morality_dimensions(interviews, [selected[method] for method in ['Dictionary', 'LDA', 'SBERT', 'NLI', 'GPT', 'DeepSeek']] + ['crowd'], waves or TEST_WAVES, n_bootstraps=0)[MORALITY_ORIGIN].round(2)
    elif name == '3':
        result = misclassifications(interviews, 'Theistic', 'fp', waves or TEST_WAVES)
    elif name == '4':
        result = misclassifications(interviews, 'Consequentialist', 'fn', waves or TEST_WAVES)
    elif name == 'llm-inputs':
        result = evaluate_morality_dimensions(interviews, ['deepseek_bin', 'deepseek_resp_bin', 'deepseek_sum_bin', 'chatgpt_bin', 'chatgpt_resp_bin', 'chatgpt_sum_bin', 'chatgpt_bin_3.5'], waves or DEVELOPMENT_WAVES).round(2)
    elif name == 'reliability':
        result = coder_reliability(interviews)
    elif name == 'coders':
        selected = select_configurations(interviews)
        result = compare_with_coders(interviews, [selected[method] for method in ['DeepSeek', 'GPT', 'NLI', 'Dictionary']] + ['crowd'])
    elif name in ['5', 'A2', 'A3', 'A8', 'A9']:
        result, descriptives = regression_table(interviews, 'chatgpt_bin' if name == 'A9' else 'deepseek_bin', ['network', 'religion', 'demographics', 'all'])
        result = descriptives['Predictions' if name == 'A2' else 'Controls'] if name in ['A2', 'A3'] else result.loc[:'Previous Behavior'] if name == '5' else result
    elif name in ['A4', 'A5', 'A6', 'A7']:
        result, _ = regression_table(interviews, 'deepseek_bin', [{'A4': 'network', 'A5': 'religion', 'A6': 'demographics', 'A7': 'all'}[name]])
    if name in ['2', 'llm-inputs']:
        result.index = [re.sub(r'[$}]', '', i).replace('_{', ' ') for i in result.index]
    display(result)
    result.to_csv(file)
    print('Saved', file)
    return result


#Predict Survey and Oral Behavior based on Morality Origin
#Returns the coefficient table (columns: without and with controls per outcome) and the descriptive statistics of
#the controls and of the outcomes (Appendix Tables A2-A3)
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
    descriptives = {}
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
            descriptives['Controls'] = stats
        
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
            descriptives['Predictions'] = stats
        
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
    return results, descriptives


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detecting moral schemas in interview transcripts: annotation, figures and tables (see README).')
    commands = parser.add_subparsers(dest='command', required=True)

    annotate_parser = commands.add_parser('annotate', help='annotate interviews with a model')
    annotate_parser.add_argument('--model', required=True, help="'wc' (dictionary), 'lda' (guided LDA), 'sbert', 'nli_quant' (NLI) or an LLM of LLM_REGISTRY (e.g. 'deepseek_bin')")
    annotate_parser.add_argument('--waves', type=int, nargs='+', default=[1, 2, 3], help='interview waves to annotate')
    annotate_parser.add_argument('--input', nargs='+', default=['Morality Text'], choices=list(INPUT_SUFFIX), help='full excerpt (Morality Text), respondent answers only (Morality Response) or summary (Morality Summary)')
    annotate_parser.add_argument('--source', default='transcripts', choices=['transcripts', 'dump'], help='raw transcripts (data/interviews/waves) or the local text dump of the excerpts')
    annotate_parser.add_argument('--gold', help='optional CSV with gold labels to evaluate the annotations')
    annotate_parser.add_argument('--local-only', action='store_true', help='refuse LLMs that send the transcripts to a third party')

    figure_parser = commands.add_parser('figure', help='reproduce a figure of the paper')
    figure_parser.add_argument('number', type=int, choices=[1, 2, 3, 4])
    figure_parser.add_argument('--waves', nargs='+', help="override the evaluation waves, e.g. --waves 'Wave 1'")

    table_parser = commands.add_parser('table', help='reproduce a table of the paper')
    table_parser.add_argument('name', choices=['2', '3', '4', '5', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'llm-inputs', 'reliability', 'coders'])
    table_parser.add_argument('--waves', nargs='+', help="override the evaluation waves, e.g. --waves 'Wave 1'")

    crowd_parser = commands.add_parser('crowd', help='crowd labeling of a wave: prepare the task or add the collected labels')
    crowd_parser.add_argument('--wave', type=int, default=3)
    crowd_parser.add_argument('--labels', help='CloudResearch export with the collected labels (omit to prepare the task)')

    args = parser.parse_args()

    if args.command == 'annotate':
        if args.local_only and not LLM_REGISTRY.get(args.model, {'local': True}).get('local', False):
            parser.error(args.model + ' sends the transcripts to a third-party API')
        annotate(args.model, args.waves, args.input, args.source, args.gold)
    elif args.command == 'figure':
        figure(args.number, prepare_data(), args.waves)
    elif args.command == 'table':
        table(args.name, prepare_data(), args.waves)
    elif args.command == 'crowd':
        add_crowd_labels(args.labels, args.wave) if args.labels else prepare_crowd_task(load_text_dump(), args.wave)
