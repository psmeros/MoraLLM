import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from __init__ import *
from sklearn.metrics import log_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import ZeroShotClassificationExplainer

from preprocessing.constants import MORALITY_ORIGIN, CODERS
from preprocessing.metadata_parser import merge_codings

#Plot cross-entropy loss for all models
def plot_cross_entropy_loss(models = ['lg', 'bert', 'bart', 'entail']):
    #Compute losses
    losses = []
    for model in models:
        interviews = pd.read_pickle('data/cache/morality_embeddings_'+model+'.pkl')
        interviews = interviews[interviews['Wave'].isin([1,3])]
        interviews = merge_codings(interviews)

        for mo in MORALITY_ORIGIN:
            interviews[mo + '_All Codings'] = interviews[mo + '_' + CODERS[0]] | interviews[mo + '_' + CODERS[1]]
            interviews[mo + '_Common Codings'] = interviews[mo + '_' + CODERS[0]] & interviews[mo + '_' + CODERS[1]]

        for c in ['All Codings', CODERS[1], CODERS[0], 'Common Codings']:
            losses.append({'Model': model, 'Codings': c, 'Loss': pd.Series({mo: log_loss(interviews[mo + '_' + c].astype(int), interviews[mo]) for mo in MORALITY_ORIGIN}).mean()})
    losses = pd.DataFrame(losses)
    baseline_loss = pd.Series({mo: log_loss(interviews[mo + '_' + CODERS[1]].astype(int), interviews[mo + '_' + CODERS[0]].astype(int)) for mo in MORALITY_ORIGIN}).sort_values()

    #Plot model comparison
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(losses, x = 'Loss', y = 'Model', hue='Codings', palette='Set2')
    plt.axvline(x=baseline_loss.mean(), linestyle='--', linewidth=4, label='Coders Agreement')
    plt.axvline(x=0.4, linestyle='--', linewidth=2, color='grey')
    plt.axvline(x=0.9, linestyle='--', linewidth=2, color='grey')
    plt.xlabel('Cross-Entropy Loss')
    plt.xscale('log')
    plt.xticks([0.4, 0.9, 6], ['0.4', '0.9', '6'])
    plt.xlim(0.1, 15)
    plt.yticks([0, 1, 2, 3], ['SpaCy', 'Bert', 'Bart', 'Entailment'])
    plt.title('Model Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(1.8, 1.03))
    plt.savefig('data/plots/evaluation-model_comparison.png', bbox_inches='tight')
    plt.show()

    #Plot coders agreement
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x = baseline_loss.values, y = baseline_loss.index, palette='Blues_d')
    plt.xticks([0, 5, 10, 15])
    plt.xlabel('Cross-Entropy Loss')
    plt.ylabel('Morality Origin')
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
    config = [1]

    for c in config:
        if c == 1:
            plot_cross_entropy_loss()
        if c == 2:
            interviews = pd.read_pickle('data/cache/morality_embeddings_entail.pkl')
            interviews = interviews[interviews['Wave'] == 1]
            interviews = merge_codings(interviews)
            explain_entailment(interviews)
        if c == 3:
            interviews = pd.read_pickle('data/cache/temporal_morality_embeddings_entail.pkl')
            plot_morality_evolution(interviews)