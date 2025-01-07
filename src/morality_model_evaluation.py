import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from IPython.display import display

from src.helpers import CODED_WAVES, CODERS, MORALITY_ESTIMATORS, MORALITY_ORIGIN
from src.parser import merge_codings, prepare_data


#Plot mean-squared error for all models
def plot_model_evaluation(models):

    #Prepare data
    interviews = prepare_data(models, extend_dataset=True)

    data = pd.concat([pd.DataFrame(interviews[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN for model in models + ['gold']]].values, columns=[mo + '_' + model for mo in MORALITY_ORIGIN for model in models + ['gold']]) for wave in CODED_WAVES]).dropna()
    data[[mo + '_gold' for mo in MORALITY_ORIGIN]] = data[[mo + '_gold' for mo in MORALITY_ORIGIN]].astype(int)
    weights = pd.Series((data[[mo + '_gold' for mo in MORALITY_ORIGIN]].sum()/data[[mo + '_gold' for mo in MORALITY_ORIGIN]].sum().sum()).values, index=MORALITY_ORIGIN)

    #Compute coders agreement
    codings = merge_codings(None, return_codings=True)
    coder_A_labels = pd.DataFrame(codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).astype(int).fillna(0)
    coder_B_labels = pd.DataFrame(codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).astype(int).fillna(0)
    coders_agreement = (pd.Series({mo:roc_auc_score(coder_A_labels[mo], coder_B_labels[mo]) for mo in MORALITY_ORIGIN}) * weights).sum()

    scores = []
    for model in models:
        score = pd.DataFrame([{mo:roc_auc_score(data[mo + '_gold'], data[mo + '_' + model], average='weighted') for mo in MORALITY_ORIGIN}])
        score['Model'] = {'lda':'SeededLDA', 'sbert':'SBERT', 'Model':'MoraLLM', 'chatgpt_prob':'GPT-4.0-Prob', 'chatgpt_bool':'GPT-4.0-Bin'}.get(model, model)
        scores.append(round(score, 2))
    scores = pd.concat(scores, ignore_index=True).iloc[::-1]
    display(scores.set_index('Model'))
    scores['AUC Score'] = (scores[MORALITY_ORIGIN] * weights).sum(axis=1).round(2)
    display(scores.set_index('Model')[['AUC Score']])
    
    #Plot model comparison
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    sns.barplot(data=scores, y='Model', hue='Model', x='AUC Score', palette='pink_r')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # tick = f1s[MORALITY_ORIGIN].sum(axis=1).max()
    # plt.axvline(x=tick, linestyle=':', linewidth=1.5, color='grey')
    plt.axvline(x=coders_agreement, linestyle='--', linewidth=1.5, color='grey', label='Annotators Agreement')
    plt.xlabel('Weighted AUC Score')
    plt.ylabel('')
    # plt.xticks([coders_agreement, tick], [str(round(coders_agreement, 2)).replace('0.', '.'), str(round(tick, 2)).replace('0.', '.')])
    plt.legend(bbox_to_anchor=(1, 1.03)).set_frame_on(False)
    plt.title('Model Evaluation')
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
    config = [4]
    
    for c in config:
        if c == 1:
            models = ['chatgpt_bin', 'chatgpt_quant', 'chatgpt_sum_bin', 'chatgpt_sum_quant', 'nli_bin', 'nli_quant', 'nli_sum_bin', 'nli_sum_quant']
            plot_model_evaluation(models)
        elif c == 2:
            plot_coders_agreement()
        elif c == 3:
            model = 'entail_ml'
            plot_ecdf(model)
        elif c == 4:
            plot_morality_distinction()