import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from sklearn.metrics import cohen_kappa_score, f1_score
from IPython.display import display

from src.helpers import CODED_WAVES, CODERS, MORALITY_ESTIMATORS, MORALITY_ORIGIN
from src.parser import merge_codings, prepare_data


#Plot mean-squared error for all models
def plot_model_evaluation(models):

    #Prepare data
    interviews = pd.read_pickle('data/cache/morality_model-entail_ml_explained.pkl')
    interviews = prepare_data(interviews, extend_dataset=True)

    data = pd.concat([pd.DataFrame(interviews[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN for model in models + ['gold']]].values, columns=[mo + '_' + model for mo in MORALITY_ORIGIN for model in models + ['gold']]) for wave in CODED_WAVES]).dropna()
    data[[mo + '_gold' for mo in MORALITY_ORIGIN]] = data[[mo + '_gold' for mo in MORALITY_ORIGIN]].astype(int)
    weights = pd.Series((data[[mo + '_gold' for mo in MORALITY_ORIGIN]].sum()/data[[mo + '_gold' for mo in MORALITY_ORIGIN]].sum().sum()).values, index=MORALITY_ORIGIN)


    for threshold in ['.1']:
        data[[mo + '_' + model + threshold for mo in MORALITY_ORIGIN for model in models]] = (data[[mo + '_' + model for mo in MORALITY_ORIGIN for model in models]] > float(threshold)).astype(int)

    #Compute coders agreement
    codings = merge_codings(None, return_codings=True)
    coder_A_labels = pd.DataFrame(codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).astype(int).fillna(0)
    coder_B_labels = pd.DataFrame(codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).astype(int).fillna(0)
    coders_agreement = (pd.Series({mo:f1_score(coder_A_labels[mo], coder_B_labels[mo]) for mo in MORALITY_ORIGIN}) * weights).sum()

    #Compute mean squared error for all models
    losses = []
    for model in models:
        if model != 'chatgpt_bool':
            for threshold in ['.1']:
                loss = pd.DataFrame([{mo:f1_score(data[mo + '_gold'], data[mo + '_' + model  + threshold]) for mo in MORALITY_ORIGIN}])
                loss['Model'] = {'lda':'SeededLDA', 'sbert':'SBERT', 'Model':'MoraLLM', 'chatgpt_prob':'GPT-4.0-Prob'}.get(model, model) + ' (' + threshold + ')'
                losses.append(loss)
        elif model == 'chatgpt_bool':
            loss = pd.DataFrame([{mo:f1_score(data[mo + '_gold'], data[mo + '_' + model]) for mo in MORALITY_ORIGIN}])
            loss['Model'] = {'chatgpt_bool':'GPT-4.0-Bin'}.get(model, model)
            losses.append(loss)

    losses = pd.concat(losses, ignore_index=True).iloc[::-1]
    display(losses.set_index('Model'))
    losses[MORALITY_ORIGIN] = losses[MORALITY_ORIGIN] * weights

    #Plot model comparison
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    losses.plot(kind='barh', x = 'Model', stacked=True, color=list(sns.color_palette('Set2')))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    tick = losses[MORALITY_ORIGIN].sum(axis=1).max()
    plt.axvline(x=tick, linestyle=':', linewidth=1.5, color='grey')
    plt.axvline(x=coders_agreement, linestyle='--', linewidth=1.5, color='grey', label='Annotators Agreement')
    plt.xlabel('F1 Score')
    plt.ylabel('')
    plt.xticks([coders_agreement, tick], [str(round(coders_agreement, 2)).replace('0.', '.'), str(round(tick, 2)).replace('0.', '.')])
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

if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    
    for c in config:
        if c == 1:
            models = ['lda', 'chatgpt_prob', 'chatgpt_bool', 'Model']
            plot_model_evaluation(models)
        elif c == 2:
            plot_coders_agreement()
        elif c == 3:
            model = 'entail_ml'
            plot_ecdf(model)