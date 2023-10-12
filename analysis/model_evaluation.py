import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.preprocessing import minmax_scale

from preprocessing.constants import CODERS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings


#Plot mean-squared error for all models
def plot_model_comparison(codings, models):

    #Compute golden labels
    coder_A_labels = codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].rename(columns={mo + '_' + CODERS[0]:mo for mo in MORALITY_ORIGIN})
    coder_B_labels = codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].rename(columns={mo + '_' + CODERS[1]:mo for mo in MORALITY_ORIGIN})
    golden_labels = coder_A_labels.astype(int) + coder_B_labels.astype(int)
    
    #Transform labels to probabilities
    golden_labels = golden_labels.div(golden_labels.sum(axis=1).apply(lambda x: 1 if x == 0 else x), axis=0)
    coder_A_labels = coder_A_labels.div(coder_A_labels.sum(axis=1).apply(lambda x: 1 if x == 0 else x), axis=0)
    coder_B_labels = coder_B_labels.div(coder_B_labels.sum(axis=1).apply(lambda x: 1 if x == 0 else x), axis=0)

    #Compute coders agreement and individual loss
    coders_agreement = pd.Series({mo:mean_squared_error(coder_A_labels[mo], coder_B_labels[mo]) for mo in MORALITY_ORIGIN}).sum()
    coder_A_loss = pd.Series({mo:mean_squared_error(golden_labels[mo], coder_A_labels[mo]) for mo in MORALITY_ORIGIN}).sum()
    coder_B_loss = pd.Series({mo:mean_squared_error(golden_labels[mo], coder_B_labels[mo]) for mo in MORALITY_ORIGIN}).sum()

    #Compute mean squared error for all models
    losses = []
    for model in models:
        if model == 'baseline':
            origin_prior = golden_labels.sum()/golden_labels.sum().sum()
            interviews = pd.DataFrame([origin_prior.tolist()] * len(codings), columns=MORALITY_ORIGIN).join(codings['Wave'])
        elif model == 'min_max':
            interviews = pd.read_pickle('data/cache/morality_model-entail.pkl')
            interviews = pd.DataFrame(minmax_scale(interviews[MORALITY_ORIGIN]), columns=MORALITY_ORIGIN).join(interviews['Wave'])
        elif model == 'one_hot':
            interviews = pd.read_pickle('data/cache/morality_model-entail.pkl')
            interviews = pd.DataFrame(interviews[MORALITY_ORIGIN].apply(lambda l: pd.Series({mo:1 if x == l.max() else 0 for mo,x in zip(MORALITY_ORIGIN, l)}), axis=1)).join(interviews['Wave'])
        else:
            interviews = pd.read_pickle('data/cache/morality_model-'+model+'.pkl')
        
        interviews = interviews[interviews['Wave'].isin([1,3])]
        loss = pd.DataFrame([{mo:mean_squared_error(golden_labels[mo], interviews[mo]) for mo in MORALITY_ORIGIN}])
        loss['Model'] = model
        losses.append(loss)

    losses = pd.concat(losses, ignore_index=True).iloc[::-1]
    losses['Model'] = losses['Model'].replace({'lg':'SpaCy (Labels)', 'bert':'BERT (Labels)', 'bart':'BART (Labels)', 'entail':'Entailment (Labels)', 'entail_explained':'Entailment (Labels+Notes)', 'top':'Entailment (Labels+Notes+Distro)', 'baseline':'Naive (Distro)'})

    #Plot model comparison
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    losses.plot(kind='barh', x = 'Model', stacked=True, colormap='Set2')
    line_1 = round(coder_A_loss, 2)
    line_2 = round(coder_B_loss, 2)
    line_3 = round(coders_agreement, 2)
    min_loss = round(losses[MORALITY_ORIGIN].sum(axis=1).min(), 2)
    plt.axvline(x=line_1, linestyle='-.', linewidth=4, color='indianred', label='Coder A Error')
    plt.axvline(x=line_2, linestyle=':', linewidth=4, color='indianred', label='Coder B Error')
    plt.axvline(x=line_3, linestyle='-', linewidth=4, color='indianred', label='Coders Disagreement')
    plt.axvline(x=min_loss, linestyle='--', linewidth=1, color='indianred')
    plt.xlabel('Mean Squared Error')
    plt.xticks([line_1, line_2, line_3, min_loss], [str(line_1)[1:], str(line_2)[1:], str(line_3)[1:], str(min_loss)[1:]])
    plt.title('Model Comparison')
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
        for mo_1 in range(len(MORALITY_ORIGIN)):
            heatmap[mo_A, mo_1] = cohen_kappa_score(coder_A[mo_A], coder_B[mo_1])
    heatmap = pd.DataFrame(heatmap, index=MORALITY_ORIGIN, columns=MORALITY_ORIGIN)

    #Plot coders agreement
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(heatmap, cmap = sns.color_palette('PuBuGn', n_colors=6), vmin=-0.2, vmax=1)
    plt.ylabel('Coder A')
    plt.xlabel('Coder B')
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([n/10 for n in range(-1, 10, 2)]) 
    colorbar.set_ticklabels(['Poor', 'Slight', 'Fair', 'Moderate', 'Substantial', 'Perfect'])

    plt.title('Coders Agreement')
    plt.savefig('data/plots/evaluation-coders_agreement.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    models = ['baseline', 'lg', 'bert', 'bart', 'entail', 'entail_explained', 'top']
    codings = pd.read_pickle('data/cache/morality_model-top.pkl')[['Wave', 'Interview Code']]
    codings = codings[codings['Wave'].isin([1,3])]
    codings = merge_codings(codings)

    for c in config:
        if c == 1:
            plot_model_comparison(codings, models)
        elif c == 2:
            plot_coders_agreement(codings)