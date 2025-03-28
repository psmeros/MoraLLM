import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from __init__ import *
from sklearn.metrics import cohen_kappa_score, f1_score
from IPython.display import display

from src.helpers import CODERS, MORALITY_ORIGIN
from src.parser import merge_codings, prepare_data


#Plot mean-squared error for all models
def plot_model_evaluation(models, evaluation_waves, n_bootstraps, human_evaluation):
    #Prepare data
    interviews = prepare_data(models, extend_dataset=True)
    interviews = pd.concat([pd.DataFrame(interviews[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN for model in models + ['gold', 'crowd'] + CODERS]].values, columns=[mo + '_' + model for mo in MORALITY_ORIGIN for model in models + ['gold', 'crowd'] + CODERS]) for wave in evaluation_waves]).dropna()
    print('Evaluation data size', len(interviews))

    scores = []
    #Bootstrapping
    for i in range(n_bootstraps):
        indices = resample(range(len(interviews)), replace=True, random_state=42 + i)
        data = interviews.iloc[indices]

        if human_evaluation:
            score = pd.DataFrame([{mo:f1_score(data[mo + '_gold'], data[mo + '_crowd'], average='weighted') for mo in MORALITY_ORIGIN}])
            score['Model'] = 'Crowdworkers'
            scores.append(round(score, 2))
            # score = pd.DataFrame(pd.DataFrame([{mo:f1_score(data[mo + '_gold'], data[mo + '_' + coder], average='weighted') for mo in MORALITY_ORIGIN} for coder in CODERS]).mean()).T
            # score['Model'] = 'Experts'
            # scores.append(round(score, 2))
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
    # plt.axvline(x=coders_agreement, linestyle='--', linewidth=1.5, color='grey', label='')
    plt.xlabel('Weighted F1 Score')
    plt.ylabel('')
    # plt.title('Model Evaluation')
    plt.savefig('data/plots/fig-model_comparison.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    
    for c in config:
        if c == 1:
            models = ['deepseek_bin', 'deepseek_bin_dto1', 'deepseek_bin_cto1', 'deepseek_bin_rto1', 'deepseek_bin_to1', 'deepseek_bin_toa', 'chatgpt_bin', 'chatgpt_bin_dto1', 'chatgpt_bin_cto1', 'chatgpt_bin_rto1', 'chatgpt_bin_to1', 'chatgpt_bin_toa']
            # models = ['deepseek_bin', 'deepseek_bin_ar', 'deepseek_bin_nt', 'chatgpt_bin', 'chatgpt_bin_ar', 'chatgpt_bin_nt']
            # models = ['nli_sum_bin', 'nli_resp_bin', 'nli_bin', 'sbert_sum_bin', 'sbert_resp_bin', 'sbert_bin', 'lda_sum_bin', 'lda_resp_bin', 'lda_bin', 'wc_sum_bin', 'wc_resp_bin', 'wc_bin']
            # models = ['deepseek_bin', 'chatgpt_bin', 'nli_sum_bin', 'sbert_resp_bin', 'lda_sum_bin', 'wc_bin']
            evaluation_waves = ['Wave 1']
            human_evaluation = False
            n_bootstraps = 10
            plot_model_evaluation(models=models, evaluation_waves=evaluation_waves, n_bootstraps=n_bootstraps, human_evaluation=human_evaluation)