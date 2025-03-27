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
            score['Model'] = {'wc_bin':'$Dictionary$', 'wc_sum_bin':'$Dictionary_{Σ}$', 'wc_resp_bin':'$Dictionary_{R}$', 'lda_bin':'$LDA$', 'lda_sum_bin':'$LDA_{Σ}$', 'lda_resp_bin':'$LDA_{R}$', 'sbert_bin':'$SBERT$', 'sbert_resp_bin':'$SBERT_{R}$', 'sbert_sum_bin':'$SBERT_{Σ}$', 'nli_bin':'$NLI$', 'nli_resp_bin':'$NLI_{R}$', 'nli_sum_bin':'$NLI_{Σ}$', 'chatgpt_bin':'$GPT$', 'chatgpt_resp_bin':'$GPT_{R}$', 'chatgpt_sum_bin':'$GPT_{Σ}$', 'chatgpt_bin_notags':'$GPT_{NT}$', 'chatgpt_bin_nodescription':'$GPT_{ND}$', 'chatgpt_bin_nointro':'$GPT_{NI}$', 'chatgpt_bin_allonce':'$GPT_{AO}$', 'chatgpt_bin_3.5':'$GPT_{3.5}$', 'chatgpt_bin_nodistinction':'$GPT_{NR}$', 'chatgpt_bin_interviewers':'$GPT_{I}$', 'chatgpt_bin_toa':'$GPT_{TOA}$', 'chatgpt_bin_to1':'$GPT_{TO1}$', 'chatgpt_bin_rto1':'$GPT_{RTO1}$', 'chatgpt_bin_cto1':'$GPT_{CTO1}$', 'chatgpt_bin_dto1':'$GPT_{DTO1}$', 'deepseek_bin':'$DeepSeek$', 'deepseek_bin_notags':'$DeepSeek_{NT}$', 'deepseek_bin_nodistinction':'$DeepSeek_{NR}$', 'deepseek_bin_nodescription':'$DeepSeek_{ND}$', 'deepseek_bin_nointro':'$DeepSeek_{NI}$', 'deepseek_bin_allonce':'$DeepSeek_{AO}$', 'deepseek_resp_bin':'$DeepSeek_{R}$', 'deepseek_sum_bin':'$DeepSeek_{Σ}$', 'deepseek_bin_toa':'$DeepSeek_{TOA}$', 'deepseek_bin_to1':'$DeepSeek_{TO1}$', 'deepseek_bin_rto1':'$DeepSeek_{RTO1}$', 'deepseek_bin_cto1':'$DeepSeek_{CTO1}$', 'deepseek_bin_dto1':'$DeepSeek_{DTO1}$'}.get(model, model)
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

#Plot coders agreement using Cohen's Kappa
def plot_coders_agreement():
    #Prepare data
    codings = merge_codings(None, return_codings=True)

    #Prepare heatmap
    coder_A = codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].astype(int).values.T
    coder_B = codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].astype(int).values.T
    heatmap = np.zeros((len(MORALITY_ORIGIN), len(MORALITY_ORIGIN)))
    
    for mo_A in range(len(MORALITY_ORIGIN)):
        for mo_B in range(len(MORALITY_ORIGIN)):
            heatmap[mo_A, mo_B] = cohen_kappa_score(coder_A[mo_A], coder_B[mo_B])
    heatmap = pd.DataFrame(heatmap, index=['Intuitive', 'Conseq.', 'Social', 'Theistic'], columns=['Intuitive', 'Conseq.', 'Social', 'Theistic'])

    #Plot coders agreement
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(heatmap, cmap = sns.color_palette('pink_r', n_colors=4), square=True, cbar_kws={'shrink': .8}, vmin=-0.2, vmax=1)
    plt.ylabel('')
    plt.xlabel('')
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-.05, .25, .55, .85])
    colorbar.set_ticklabels(['Poor', 'Slight', 'Moderate', 'Perfect'])
    plt.title('Cohen\'s Kappa Agreement between Annotators')
    plt.savefig('data/plots/fig-coders_agreement.png', bbox_inches='tight')
    plt.show()

#Plot morality distinction on synthetic data
def plot_morality_distinction():
    #Prepare Data
    data = pd.read_pickle('data/cache/synthetic_data.pkl')
    data['Distinction'] = data['Distinction'] * 100
    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=data, x='Distinction', y='Morality', hue='Morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Strong-Weak Morality Distinction', x=0.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-100, 100))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-synthetic_distinction.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    
    for c in config:
        if c == 1:
            models = ['deepseek_bin', 'deepseek_bin_toa', 'chatgpt_bin', 'chatgpt_bin_toa']
            # models = ['nli_sum_bin', 'nli_resp_bin', 'nli_bin', 'sbert_sum_bin', 'sbert_resp_bin', 'sbert_bin', 'lda_sum_bin', 'lda_resp_bin', 'lda_bin', 'wc_sum_bin', 'wc_resp_bin', 'wc_bin']
            # models = ['deepseek_bin', 'chatgpt_bin', 'nli_sum_bin', 'sbert_resp_bin', 'lda_sum_bin', 'wc_bin']
            evaluation_waves = ['Wave 1']
            human_evaluation = False
            n_bootstraps = 10
            plot_model_evaluation(models=models, evaluation_waves=evaluation_waves, n_bootstraps=n_bootstraps, human_evaluation=human_evaluation)
        elif c == 2:
            plot_coders_agreement()
        elif c == 3:
            plot_morality_distinction()