import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import ZeroShotClassificationExplainer

from preprocessing.constants import CODERS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings


#Plot mean-squared error for all models
def plot_model_comparison(interviews, models):

    #Coders agreement
    coders_agreement = pd.Series({mo:mean_squared_error(interviews[mo + '_' + CODERS[0]].astype(int), interviews[mo + '_' + CODERS[1]].astype(int)) for mo in MORALITY_ORIGIN}).sum()
    golden_labels = interviews.apply(lambda i: pd.Series(i[mo + '_' + CODERS[0]] & i[mo + '_' + CODERS[1]] for mo in MORALITY_ORIGIN), axis=1).rename(columns={i:mo for i, mo in enumerate(MORALITY_ORIGIN)})
    coder_A_loss = pd.Series({mo:mean_squared_error(golden_labels[mo].astype(int), interviews[mo + '_' + CODERS[0]].astype(int)) for mo in MORALITY_ORIGIN}).sum()
    coder_B_loss = pd.Series({mo:mean_squared_error(golden_labels[mo].astype(int), interviews[mo + '_' + CODERS[1]].astype(int)) for mo in MORALITY_ORIGIN}).sum()
    
    #Baseline prior classifier
    origin_prior = golden_labels.sum()/golden_labels.sum().sum()
    prior_classifier = pd.DataFrame([origin_prior.tolist()] * len(interviews), columns=MORALITY_ORIGIN)
        
    #Compute mean squared error
    loss = pd.DataFrame([{mo:mean_squared_error(golden_labels[mo].astype(int), prior_classifier[mo]) for mo in MORALITY_ORIGIN}])
    loss['Model'] = 'Baseline (Prior)'
    losses = [loss]

    for model in models:
        interviews = pd.read_pickle('data/cache/morality_embeddings_'+model+'.pkl')
        interviews = interviews[interviews['Wave'].isin([1,3])]

        loss = pd.DataFrame([{mo:mean_squared_error(golden_labels[mo].astype(int), interviews[mo]) for mo in MORALITY_ORIGIN}])
        loss['Model'] = model
        losses = [loss] + losses

    losses = pd.concat(losses, ignore_index=True)
    losses['Model'] = losses['Model'].replace({'lg':'SpaCy', 'bert':'BERT', 'bart':'BART', 'entail':'Entailment', 'lg_multilabel':'SpaCy (ML)', 'bert_multilabel':'BERT (ML)', 'bart_multilabel':'BART (ML)', 'entail_multilabel':'Entailment (ML)'})

    #Plot model comparison
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    losses.plot(kind='barh', x = 'Model', stacked=True, colormap='Set2')
    line_1 = round(coder_A_loss, 1)
    line_2 = round(coder_B_loss, 1)
    line_3 = round(coders_agreement, 1)
    plt.axvline(x=line_1, linestyle='-.', linewidth=4, color='indianred', label='Coder A Error')
    plt.axvline(x=line_2, linestyle=':', linewidth=4, color='indianred', label='Coder B Error')
    plt.axvline(x=line_3, linestyle='-', linewidth=4, color='indianred', label='Coders Disagreement')
    plt.xlabel('Mean Squared Error')
    plt.xticks([line_1, line_2, line_3], [str(line_1), str(line_2), str(line_3)])
    plt.xlim(.0, 2.2)
    plt.title('Model Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(1.65, 1.03), fontsize='small')
    plt.savefig('data/plots/evaluation-model_comparison.png', bbox_inches='tight')
    plt.show()

#Plot coders agreement using Cohen's Kappa
def plot_coders_agreement(interviews):
    #Prepare heatmap
    coder_0 = interviews[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].astype(int).values.T
    coder_1 = interviews[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].astype(int).values.T
    heatmap = np.zeros((len(MORALITY_ORIGIN), len(MORALITY_ORIGIN)))
    
    for mo_0 in range(len(MORALITY_ORIGIN)):
        for mo_1 in range(len(MORALITY_ORIGIN)):
            heatmap[mo_0, mo_1] = cohen_kappa_score(coder_0[mo_0], coder_1[mo_1])
    heatmap = pd.DataFrame(heatmap, index=MORALITY_ORIGIN, columns=MORALITY_ORIGIN)

    #Plot coders agreement
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(heatmap, cmap = sns.color_palette('PuBuGn', n_colors=6), vmin=-0.2, vmax=1)
    plt.ylabel(CODERS[0])
    plt.xlabel(CODERS[1])
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([n/10 for n in range(-1, 10, 2)]) 
    colorbar.set_ticklabels(['Poor', 'Slight', 'Fair', 'Moderate', 'Substantial', 'Perfect'])

    plt.title('Coders Agreement')
    plt.savefig('data/plots/evaluation-coders_agreement.png', bbox_inches='tight')
    plt.show()

#Explain word-level attention for zero-shot models
def explain_entailment(interviews):
    pairs = [(interviews.iloc[interviews[mo + '_x'].idxmax()]['Morality_Origin'], [mo]) for mo in MORALITY_ORIGIN]

    model_name = 'cross-encoder/nli-deberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    zero_shot_explainer = ZeroShotClassificationExplainer(model, tokenizer)

    for text, labels in pairs:
        zero_shot_explainer(text=text, hypothesis_template='The morality origin is {}.',labels=labels)
        zero_shot_explainer.visualize('data/misc/zero_shot.html')

#Plot morality evolution over waves
def plot_morality_evolution(interviews):
    #Merge waves
    wave_1 = interviews[['Wave 1:' + mo for mo in MORALITY_ORIGIN]].rename(columns={'Wave 1:' + mo: mo for mo in MORALITY_ORIGIN})
    wave_2 = interviews[['Wave 2:' + mo for mo in MORALITY_ORIGIN]].rename(columns={'Wave 2:' + mo: mo for mo in MORALITY_ORIGIN})
    wave_3 = interviews[['Wave 3:' + mo for mo in MORALITY_ORIGIN]].rename(columns={'Wave 3:' + mo: mo for mo in MORALITY_ORIGIN})
    wave_1['Wave'] = 1
    wave_2['Wave'] = 2
    wave_3['Wave'] = 3
    interviews = pd.concat([wave_1, wave_2, wave_3])

    #Prepare for plotting
    interviews = pd.melt(interviews, id_vars=['Wave'], value_vars=MORALITY_ORIGIN, var_name='Morality Origin', value_name='Value')
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.lineplot(data=interviews, y='Value', x='Wave', hue='Morality Origin', linewidth=4, palette='Set2')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.ylabel('')
    plt.xticks([1,2,3])
    plt.title('Morality Evolution')
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1.03))
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.savefig('data/plots/evaluation-morality_evolution.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [1,2]
    models = ['entail-explained']
    interviews = pd.read_pickle('data/cache/morality_embeddings_entail-explained.pkl')
    interviews = interviews[interviews['Wave'].isin([1,3])]
    interviews = merge_codings(interviews)
    temporal_interviews = pd.read_pickle('data/cache/temporal_morality_embeddings_entail.pkl')

    for c in config:
        if c == 1:
            plot_model_comparison(interviews, models)
        elif c == 2:
            plot_coders_agreement(interviews)
        elif c == 3:
            explain_entailment(interviews)
        elif c == 4:
            plot_morality_evolution(temporal_interviews)