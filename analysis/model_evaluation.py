import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from sklearn.metrics import cohen_kappa_score, mean_squared_error

from preprocessing.constants import CODERS, MERGE_MORALITY_ORIGINS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings


#Plot mean-squared error for all models
def plot_model_evaluation(codings, models):
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
            interviews = interviews.drop(['Experience', 'Consequences', 'Family', 'Community', 'Friends', 'Media', 'Laws', 'Holy Scripture'], axis=1)

        interviews = merge_codings(interviews)[MORALITY_ORIGIN]
        loss = pd.DataFrame([{mo:mean_squared_error(golden_labels[mo], interviews[mo]) for mo in MORALITY_ORIGIN}]) * weights
        loss['Model'] = model
        losses.append(loss)

    losses = pd.concat(losses, ignore_index=True).iloc[::-1]
    losses['Model'] = losses['Model'].replace({'lg':'SpaCy (Labels)', 'bert':'BERT (Labels)', 'bart':'BART (Labels)', 'entail_ml':'Entailment (Labels)', 'entail_ml_explained':'Entailment (Labels+Notes)', 'top':'Entailment (Labels+Notes+Distro)', 'chatgpt':'GPT-3.5 (Labels)'})

    #Plot model comparison
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    losses.plot(kind='barh', x = 'Model', stacked=True, color=list(sns.color_palette('Set2')))
    min_loss = losses[MORALITY_ORIGIN].sum(axis=1).min()
    plt.axvline(x=coders_agreement, linestyle='-', linewidth=4, color='indianred', label='Coders Disagreement')
    plt.axvline(x=min_loss, linestyle='--', linewidth=1, color='grey')
    plt.xlabel('Normalized Mean Squared Error')
    plt.ylabel('')
    plt.xticks([coders_agreement, min_loss], [str(round(coders_agreement, 2))[1:], str(round(min_loss, 2))[1:]])
    plt.legend(loc='upper right', bbox_to_anchor=(1.65, 1.03), fontsize='small')
    plt.savefig('data/plots/evaluation-model_comparison.png', bbox_inches='tight')
    plt.show()

#Plot coders agreement using Cohen's Kappa
def plot_coders_agreement(codings):
    #Prepare heatmap
    coder_A = codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].astype(int).values.T
    coder_B = codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].astype(int).values.T
    heatmap = np.zeros((len(MORALITY_ORIGIN), len(MORALITY_ORIGIN)))
    
    for mo_A in range(len(MORALITY_ORIGIN)):
        for mo_B in range(len(MORALITY_ORIGIN)):
            heatmap[mo_A, mo_B] = cohen_kappa_score(coder_A[mo_A], coder_B[mo_B])
    heatmap = pd.DataFrame(heatmap, index=MORALITY_ORIGIN, columns=MORALITY_ORIGIN)

    #Plot coders agreement
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(heatmap, cmap = sns.color_palette('PuBuGn', n_colors=6), vmin=-0.2, vmax=1)
    plt.ylabel('')
    plt.xlabel('')
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([n/10 for n in range(-1, 10, 2)]) 
    colorbar.set_ticklabels(['Poor', 'Slight', 'Fair', 'Moderate', 'Substantial', 'Perfect'])
    plt.title('Cohen\'s Kappa Agreement between Coders')
    plt.savefig('data/plots/evaluation-coders_agreement.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [1,2]
    models = ['lg', 'bert', 'bart', 'chatgpt', 'entail_ml', 'entail_ml_explained', 'top']
    codings = pd.read_pickle('data/cache/morality_model-top.pkl')[['Wave', 'Interview Code']]
    codings = merge_codings(codings)

    for c in config:
        if c == 1:
            plot_model_evaluation(codings, models)
        elif c == 2:
            plot_coders_agreement(codings)