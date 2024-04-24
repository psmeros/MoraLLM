import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from sklearn.metrics import cohen_kappa_score, mean_squared_error

from preprocessing.constants import CODERS, MERGE_MORALITY_ORIGINS, MORALITY_ESTIMATORS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings


#Plot mean-squared error for all models
def plot_model_evaluation(interviews, models):
    #Prepare data
    codings = merge_codings(interviews[['Wave', 'Interview Code']])

    #Compute golden labels
    coder_A_labels = pd.DataFrame(codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).astype(int).fillna(0)
    coder_B_labels = pd.DataFrame(codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).astype(int).fillna(0)
    golden_labels = (coder_A_labels & coder_B_labels)

    #Loss weights
    weights = 1 - golden_labels.sum()/golden_labels.sum().sum()

    #Compute coders agreement
    coders_agreement = (pd.Series({mo:mean_squared_error(coder_A_labels[mo], coder_B_labels[mo]) for mo in MORALITY_ORIGIN}) * weights).sum()

    #Compute mean squared error for all models
    losses = []
    for model in models:
        interviews = pd.read_pickle('data/cache/morality_model-'+model+'.pkl')
        if model != 'top' and MERGE_MORALITY_ORIGINS:
            interviews['Intuitive'] = interviews['Experience']
            interviews['Consequentialist'] = interviews['Consequences']
            interviews['Social'] = interviews[['Family', 'Community', 'Friends']].max(axis=1)
            interviews['Theistic'] = interviews['Holy Scripture']
            interviews = interviews.drop(['Experience', 'Consequences', 'Family', 'Community', 'Friends', 'Media', 'Laws', 'Holy Scripture'], axis=1)

        interviews = merge_codings(interviews)[MORALITY_ORIGIN]
        loss = pd.DataFrame([{mo:mean_squared_error(golden_labels[mo], interviews[mo]) for mo in MORALITY_ORIGIN}]) * weights
        loss['Model'] = model
        losses.append(loss)

    losses = pd.concat(losses, ignore_index=True).iloc[::-1]
    losses['Model'] = losses['Model'].replace({'lg':'SpaCy', 'bert':'BERT', 'bart':'BART', 'entail':'Entailment', 'entail_explained':'Entailment (Labels+Notes)', 'top':'MoraLLM', 'chatgpt':'GPT-3.5'})

    #Plot model comparison
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    losses.plot(kind='barh', x = 'Model', stacked=True, color=list(sns.color_palette('Set2')))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    min_loss = losses[MORALITY_ORIGIN].sum(axis=1).min()
    plt.axvline(x=coders_agreement, linestyle='-', linewidth=4, color='indianred', label='Coders Agreement')
    plt.axvline(x=min_loss, linestyle='--', linewidth=1, color='grey')
    plt.xlabel('Normalized Mean Squared Error')
    plt.ylabel('')
    plt.xticks([coders_agreement, min_loss], [str(round(coders_agreement, 1)).replace('0.', '.'), str(round(min_loss, 1)).replace('0.', '.')])
    plt.legend(bbox_to_anchor=(1, 1.03)).set_frame_on(False)
    plt.savefig('data/plots/evaluation-model_comparison.png', bbox_inches='tight')
    plt.show()

#Plot coders agreement using Cohen's Kappa
def plot_coders_agreement(interviews):
    #Prepare data
    codings = merge_codings(interviews[['Wave', 'Interview Code']])

    #Prepare heatmap
    coder_A = codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].astype(int).values.T
    coder_B = codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].astype(int).values.T
    heatmap = np.zeros((len(MORALITY_ORIGIN), len(MORALITY_ORIGIN)))
    
    for mo_A in range(len(MORALITY_ORIGIN)):
        for mo_B in range(len(MORALITY_ORIGIN)):
            heatmap[mo_A, mo_B] = cohen_kappa_score(coder_A[mo_A], coder_B[mo_B])
    heatmap = pd.DataFrame(heatmap, index=MORALITY_ORIGIN, columns=MORALITY_ORIGIN)

    #Plot coders agreement
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=3)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(heatmap, cmap = sns.color_palette('PuBuGn', n_colors=6), square=True, cbar_kws={'shrink': .8}, vmin=-0.2, vmax=1)
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([n/10 for n in range(-1, 10, 2)]) 
    colorbar.set_ticklabels(['Poor', 'Slight', 'Fair', 'Moderate', 'Substantial', 'Perfect'])
    plt.title('Cohen\'s Kappa Agreement between Experts')
    plt.savefig('data/plots/evaluation-coders_agreement.png', bbox_inches='tight')
    plt.show()

#Show benefits of quantification by plotting ecdf
def plot_ecdf(interviews):
    #Prepare Data
    interviews = merge_codings(interviews)
    model = pd.DataFrame(interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    model['Estimator'] = 'Model'
    coders = pd.DataFrame(interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    coders['Estimator'] = 'Coders'
    interviews = pd.concat([model, coders])
    interviews = interviews.melt(id_vars=['Estimator'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.displot(data=interviews, x='Value', hue='Morality', col='Estimator', kind='ecdf', linewidth=3, aspect=.85, palette=sns.color_palette('Set2')[:len(MORALITY_ORIGIN)])
    g.set_titles('{col_name}')
    g.legend.set_title('')
    g.set_xlabels('')
    g.set_ylabels('')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    plt.savefig('data/plots/evaluation-morality_ecdf.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [1,2,3]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    
    for c in config:
        if c == 1:
            models = ['lg', 'bert', 'bart', 'chatgpt', 'top']
            plot_model_evaluation(interviews, models)
        elif c == 2:
            plot_coders_agreement(interviews)
        elif c == 3:
            plot_ecdf(interviews)