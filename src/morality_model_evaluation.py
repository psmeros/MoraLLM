import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
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

    #Compute best threshold for binarization
    best_thresholds = []
    max_f1s = []
    for mo in MORALITY_ORIGIN:
        max_f1 = 0
        best_threshold = 0
        for threshold in range(-300, 300, 5):
            slice = (scale(data[[mo + '_Model']]) > threshold/100).astype(int)
            f1 = f1_score(data[mo + '_gold'], slice, average='weighted')
            if f1 > max_f1:
                max_f1 = f1
                best_threshold = threshold/100
        best_thresholds.append(best_threshold)
        max_f1s.append(max_f1)
    print((pd.Series(max_f1s, index=MORALITY_ORIGIN) * weights).sum())
    data[[mo + '_nli_bin' for mo in MORALITY_ORIGIN]] = (scale(data[[mo + '_Model' for mo in MORALITY_ORIGIN]]) > best_thresholds).astype(int)
    models = list(map(lambda x: x.replace('Model', 'nli_bin'), models))

    #Compute coders agreement
    codings = merge_codings(None, return_codings=True)
    coder_A_labels = pd.DataFrame(codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).astype(int).fillna(0)
    coder_B_labels = pd.DataFrame(codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).astype(int).fillna(0)
    coders_agreement = (pd.Series({mo:f1_score(coder_A_labels[mo], coder_B_labels[mo], average='weighted') for mo in MORALITY_ORIGIN}) * weights).sum()

    #Compute f1 score for all models
    f1s = []
    for model in models:
        f1 = pd.DataFrame([{mo:f1_score(data[mo + '_gold'], data[mo + '_' + model], average='weighted') for mo in MORALITY_ORIGIN}])
        f1['Model'] = {'lda':'SeededLDA', 'sbert':'SBERT', 'Model':'MoraLLM', 'chatgpt_prob':'GPT-4.0-Prob', 'chatgpt_bool':'GPT-4.0-Bin'}.get(model, model)
        f1s.append(f1)

    f1s = pd.concat(f1s, ignore_index=True).iloc[::-1]
    display(f1s.set_index('Model'))
    f1s[MORALITY_ORIGIN] = f1s[MORALITY_ORIGIN] * weights

    #Plot model comparison
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    f1s.plot(kind='barh', x = 'Model', stacked=True, color=list(sns.color_palette('Set2')))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    tick = f1s[MORALITY_ORIGIN].sum(axis=1).max()
    plt.axvline(x=tick, linestyle=':', linewidth=1.5, color='grey')
    plt.axvline(x=coders_agreement, linestyle='--', linewidth=1.5, color='grey', label='Annotators Agreement')
    plt.xlabel('Weighted F1 Score')
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
            models = ['chatgpt_bin', 'chatgpt_quant', 'Model']
            plot_model_evaluation(models)
        elif c == 2:
            plot_coders_agreement()
        elif c == 3:
            model = 'entail_ml'
            plot_ecdf(model)