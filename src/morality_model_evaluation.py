import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from __init__ import *
from sklearn.metrics import cohen_kappa_score, f1_score
from IPython.display import display

from src.helpers import CODED_WAVES, CODERS, MORALITY_ESTIMATORS, MORALITY_ORIGIN
from src.parser import merge_codings, prepare_data


#Plot mean-squared error for all models
def plot_model_evaluation(models, evaluation_waves, n_bootstraps, human_evaluation, palette):
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
            score['Model'] = {'wc_bin':'$Dictionary$', 'wc_sum_bin':'$WC_{Σ}$', 'wc_resp_bin':'$WC_{R}$', 'lda_bin':'$LDA$', 'lda_sum_bin':'$LDA_{Σ}$', 'lda_resp_bin':'$LDA_{R}$', 'sbert_bin':'$SBERT$', 'sbert_resp_bin':'$SBERT_{R}$', 'sbert_sum_bin':'$SBERT_{Σ}$', 'nli_bin':'$NLI$', 'nli_resp_bin':'$NLI_{R}$', 'nli_sum_bin':'$NLI_{Σ}$', 'chatgpt_bin':'$GPT4$', 'chatgpt_resp_bin':'$GPT4_{R}$', 'chatgpt_sum_bin':'$GPT4_{Σ}$', 'chatgpt_bin_notags':'$GPT4_{NT}$', 'chatgpt_bin_3.5':'$GPT3.5_{F}$', 'chatgpt_bin_nodistinction':'$GPT4_{ND}$', 'chatgpt_bin_interviewers':'$GPT4_{I}$'}.get(model, model)
            scores.append(round(score, 2))
    scores = pd.concat(scores, ignore_index=True).iloc[::-1]
    display(scores.set_index('Model').groupby('Model', sort=False).mean().round(2))
    scores.set_index('Model').groupby('Model', sort=False).mean().round(2).to_clipboard()
    scores['score'] = (scores[MORALITY_ORIGIN]).mean(axis=1).round(2)
    display(scores.set_index('Model').groupby('Model', sort=False).mean().round(2)[['score']])
    
    #Plot model comparison
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=scores, y='Model', hue='Model', x='score', palette=palette)
    ax = plt.gca()
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

#Show benefits of quantification by plotting ecdf
def plot_ecdf(model):
    #Prepare Data
    interviews = pd.read_pickle('data/cache/morality_model-' + model + '.pkl')
    interviews = merge_codings(interviews)
    model = pd.DataFrame(interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    model['Estimator'] = 'MoraLLM'
    coders = pd.DataFrame(interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    coders['Estimator'] = 'Annotators'
    interviews = pd.concat([coders, model])
    interviews = interviews.melt(id_vars=['Estimator'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.displot(data=interviews, x='Value', hue='Morality', col='Estimator', kind='ecdf', linewidth=5, aspect=.85, palette=sns.color_palette('Set2')[:len(MORALITY_ORIGIN)])
    g.figure.suptitle('Cumulative Distribution Function', y=1.05)
    g.set_titles('{col_name}')
    g.legend.set_title('')
    g.set_xlabels('')
    g.set_ylabels('')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    plt.savefig('data/plots/fig-morality_ecdf.png', bbox_inches='tight')
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
            # models = ['chatgpt_bin_3.5', 'chatgpt_sum_bin', 'chatgpt_bin_notags', 'chatgpt_bin']
            # models = ['chatgpt_bin_interviewers', 'chatgpt_bin_nodistinction', 'chatgpt_bin_notags', 'chatgpt_bin']
            # palette = sns.color_palette('Blues', len(models))
            # models = ['chatgpt_sum_bin', 'chatgpt_resp_bin', 'chatgpt_bin', 'nli_sum_bin', 'nli_resp_bin', 'nli_bin', 'sbert_sum_bin', 'sbert_resp_bin', 'sbert_bin', 'lda_sum_bin', 'lda_resp_bin', 'lda_bin', 'wc_sum_bin', 'wc_resp_bin', 'wc_bin']
            # palette = [c for c in sns.color_palette('Blues', 5) for _ in range(3)] + sns.color_palette('Purples', 5)[4:5]*2
            models = ['chatgpt_bin', 'nli_bin', 'sbert_bin', 'lda_bin', 'wc_bin']
            palette = [c for c in sns.color_palette('Blues', 5) for _ in range(1)] + sns.color_palette('Purples', 5)[4:5]*2
            evaluation_waves = ['Wave 1']
            human_evaluation = True
            n_bootstraps = 10
            plot_model_evaluation(models=models, evaluation_waves=evaluation_waves, n_bootstraps=n_bootstraps, human_evaluation=human_evaluation, palette=palette)
        elif c == 2:
            plot_coders_agreement()
        elif c == 3:
            model = 'entail_ml'
            plot_ecdf(model)
        elif c == 4:
            plot_morality_distinction()