import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import patsy
import seaborn as sns
from scipy.stats import pearsonr
from IPython.display import display
from sklearn.preprocessing import scale
from statsmodels.regression.linear_model import OLS

from __init__ import *
from src.helpers import DEMOGRAPHICS, INCOME_RANGE, MORALITY_ORIGIN, format_pvalue
from src.parser import prepare_data

#Show benefits of quantification by plotting ecdf
def plot_ecdf(interviews, model):
    #Prepare Data
    model = pd.DataFrame(interviews[['Wave 1:' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).assign(Estimator='NLI').assign(Estimator='NLI').dropna()
    crowd = pd.DataFrame(interviews[['Wave 1:' + mo + '_' + 'crowd' for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).assign(Estimator='NLI').assign(Estimator='Crowd').dropna()
    data = pd.concat([model, crowd])
    data = data.melt(id_vars=['Estimator'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.displot(data=data, x='Value', hue='Morality', col='Estimator', kind='ecdf', linewidth=5, aspect=.85, palette=sns.color_palette('Set2')[:len(MORALITY_ORIGIN)])
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

#Plot Intuitive-Consequentialist and Social-Theistic Morality Distinction
def plot_morality_distinction(interviews, model, waves, to_latex):
    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN) for wave in waves]).reset_index(drop=True)

    #Compute Correlations
    results = pd.DataFrame(index=[mo1 + ' - ' + mo2 for i, mo1 in enumerate(MORALITY_ORIGIN) for j, mo2 in enumerate(MORALITY_ORIGIN) if i < j] + ['N'], columns=['Correlation'])
    for i in results.index[:-1]:
        results.loc[i] = format_pvalue(pearsonr(data[i.split(' - ')[0]], data[i.split(' - ')[1]]))
    results.loc['N'] = len(data)
    
    #Show Results
    print(results.to_latex()) if to_latex else display(results)
    results.to_clipboard()

    #Compute Distinction
    data = data.apply(lambda x: pd.Series([x['Intuitive'] - x['Consequentialist'], x['Social'] - x['Theistic']]), axis=1)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(data=data, x=0, y=1, kind='hex', color='rosybrown')
    g.figure.suptitle('Morality Distinction', y=1.03)
    ax = plt.gca()
    ax.xaxis.set_ticks([-1, -.5, 0, .5, 1])
    ax.yaxis.set_ticks([-1, -.5, 0, .5, 1])
    ax.set_xlabel('Intuitive-Consequentialist Distinction')
    ax.set_ylabel('Social-Theistic Distinction')
    plt.savefig('data/plots/fig-morality_distinction.png', bbox_inches='tight')
    plt.show()

def compute_decisiveness(interviews, model, waves, decisive_threshold = .5):
    #Prepare Data
    data = interviews.copy()
    decisiveness_options = ['Rigidly Decisive', 'Ambivalent', 'Rigidly Indecisive']
    decisiveness = data.apply(lambda i: pd.Series(((i[waves[0] + ':' + mo + '_' + model] >= decisive_threshold), (i[waves[1] + ':' + mo + '_' + model] >= decisive_threshold)) for mo in MORALITY_ORIGIN), axis=1).set_axis(MORALITY_ORIGIN, axis=1)
    decisiveness = decisiveness.map(lambda d: decisiveness_options[0] if d[0] and d[1] else decisiveness_options[1] if not d[0] and d[1] else decisiveness_options[1] if d[0] and not d[1] else decisiveness_options[2] if not d[0] and not d[1] else '')
    
    decisiveness = decisiveness.apply(lambda x: x.value_counts(normalize=True) * 100).T
    decisiveness = decisiveness.stack().reset_index().rename(columns={'level_0':'Morality', 'level_1':'Decisiveness', 0:'Value'})

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=3.5)
    plt.figure(figsize=(10, 10))

    sns.barplot(data=decisiveness, y='Morality', x='Value', hue='Decisiveness', order=MORALITY_ORIGIN, hue_order=decisiveness_options, palette=sns.color_palette('coolwarm', n_colors=len(decisiveness_options)))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y :.0f}%'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Crosswave Decisiveness')
    plt.legend(bbox_to_anchor=(1, 1.03)).set_frame_on(False)
    plt.savefig('data/plots/fig-decisiveness.png', bbox_inches='tight')
    plt.show()

#Compute morality evolution across waves
def plot_morality_evolution(interviews, model, waves):
    #Prepare data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN] + [wave + ':' + d for d in DEMOGRAPHICS]].values, columns=MORALITY_ORIGIN + DEMOGRAPHICS).assign(Wave=int(wave.split()[1])) for wave in waves]).dropna().reset_index(drop=True)
    data = data.melt(id_vars=['Wave']+DEMOGRAPHICS, value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    data['Value'] = pd.to_numeric(data['Value']) * 100

    data['Race'] = data['Race'].map(lambda r: {'White': 'White', 'Black': 'Other', 'Other': 'Other'}.get(r, None))
    data['Household Income'] = data['Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    for demographic in DEMOGRAPHICS:
        data[demographic] = data[demographic].map(lambda x: x + ' (N = ' + str(int(len(data[data[demographic] == x])/len(MORALITY_ORIGIN)/len(waves))) + ')')

    #Plot overall morality evolution
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    ax = sns.lineplot(data=data, x='Wave', y='Value', hue='Morality', hue_order=MORALITY_ORIGIN, seed=42, palette='Set2')
    ax.set_ylim(20, 100)
    ax.set_yticks([20, 60, 100])
    ax.set_xticks([int(w.split()[1]) for w in waves])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='', frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.suptitle('Morality Evolution over Waves', y=1)
    plt.savefig('data/plots/fig-morality_evolution_overall.png', bbox_inches='tight')
    plt.show()

    #Plot morality evolution by demographic
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=3)
    plt.figure(figsize=(20, 10))
    _, axes = plt.subplots(len(MORALITY_ORIGIN), len(DEMOGRAPHICS), figsize=(40, 10 * len(MORALITY_ORIGIN)))
    for i, morality in enumerate(MORALITY_ORIGIN):
        for j, demographic in enumerate(DEMOGRAPHICS):
            sns.lineplot(data=data[data['Morality'] == morality], x='Wave', y='Value', hue=demographic, seed=42, ax=axes[i, j], palette='Set1')
            axes[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
            axes[i, j].set_ylim(20, 100)
            axes[i, j].set_yticks([20, 60, 100])
            axes[i, j].set_xticks([int(w.split()[1]) for w in waves])
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].legend(title=demographic, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3) if i == 0 else axes[i, j].legend().set_visible(False)
            axes[i, j].set_xlabel('Wave')
            axes[i, j].set_ylabel(morality if j == 0 else '')
    plt.tight_layout()
    plt.suptitle('Morality Evolution over Waves', y=1.02)
    plt.savefig('data/plots/fig-morality_evolution_by_demographic.png', bbox_inches='tight')

#Compute morality shifts across waves
def plot_morality_shift(interviews, model, waves):
    data = interviews.copy()

    #Standardize waves separately
    data[[wave + ':' + mo + '_' + model for wave in waves for mo in MORALITY_ORIGIN]] = scale(data[[wave + ':' + mo + '_' + model for wave in waves for mo in MORALITY_ORIGIN]], with_std=False)

    #Compute morality shifts across waves
    shifts = pd.DataFrame(data[[waves[1] + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values - data[[waves[0] + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    shifts[DEMOGRAPHICS] = data[[waves[0] + ':' + d for d in DEMOGRAPHICS]].values
    shifts = shifts.dropna().melt(id_vars=DEMOGRAPHICS, value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')

    #Prepare data
    shifts['Value'] = shifts['Value'] * 100
    shifts['Race'] = shifts['Race'].map(lambda r: {'White': 'White', 'Black': 'Other', 'Other': 'Other'}.get(r, None))
    shifts['Household Income'] = shifts['Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    for demographic in DEMOGRAPHICS:
        shifts[demographic] = shifts[demographic].map(lambda x: x + ' (N = ' + str(int(len(shifts[shifts[demographic] == x])/len(MORALITY_ORIGIN))) + ')')

    #Plot overall morality shifts
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=shifts, x='Value', y='Morality', hue='Morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Morality Shift over Waves', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-25, 25))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_shift_overall.png', bbox_inches='tight')
    plt.show()

    #Plot morality shifts by demographic
    plt.figure(figsize=(20, 10))
    _, axes = plt.subplots(2, 2, figsize=(20, 10))
    for i, demographic in enumerate(DEMOGRAPHICS):
        sns.barplot(data=shifts, x='Value', y='Morality', hue=demographic, order=MORALITY_ORIGIN, dodge=0.3, ax=axes[i//2,i%2], errorbar=None, palette='Set1')
        axes[i//2,i%2].axvline(x=0, color='grey', linestyle='--', linewidth=1.5)
        axes[i//2,i%2].set_xlim(-25, 25)
        axes[i//2,i%2].set_xlabel('')
        axes[i//2,i%2].set_ylabel('')
        axes[i//2,i%2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
        axes[i//2,i%2].spines['top'].set_visible(False)
        axes[i//2,i%2].spines['right'].set_visible(False)
        axes[i//2,i%2].legend(title=demographic, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3)
    plt.tight_layout()
    plt.suptitle('Morality Shift over Waves')
    plt.savefig('data/plots/fig-morality_shift_by_demographic.png', bbox_inches='tight')

#Predict morality shifts across waves
def predict_shift(interviews, model, conf, to_latex):
    data = interviews.copy()

    #Compute morality shifts across waves
    data = pd.DataFrame(data[[waves[1] + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values - data[[waves[0] + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    data[DEMOGRAPHICS] = interviews[[waves[0] + ':' + d for d in DEMOGRAPHICS]].values
    data['Race'] = data['Race'].map(lambda r: {'White': 'White', 'Black': 'Other', 'Other': 'Other'}.get(r, None))
    data['Household Income'] = scale(data['Household Income'])
    
    #Add Reference Controls
    for attribute_name in conf['References']['Attribute Names']:
        dummies = pd.get_dummies(data[attribute_name], prefix=attribute_name, prefix_sep=' = ').astype(int)
        data = pd.concat([data, dummies], axis=1).drop(attribute_name, axis=1)
        conf['Controls'] = conf['Controls'][:conf['Controls'].index(attribute_name)] + list(dummies.columns) + conf['Controls'][conf['Controls'].index(attribute_name) + 1:]

    #Convert Data to Numeric
    data = data.apply(pd.to_numeric)

    #Drop NA and Reference Dummies
    conf['Controls'] = [c for c in conf['Controls'] if c not in [attribute_name + ' = ' + attribute_value for attribute_name, attribute_value in zip(conf['References']['Attribute Names'], conf['References']['Attribute Values'])]]
    data = data.drop([attribute_name + ' = ' + attribute_value for attribute_name, attribute_value in zip(conf['References']['Attribute Names'], conf['References']['Attribute Values'])], axis=1)
    data = data.reset_index(drop=True)

    #Define Formulas
    formulas = ['Q("' + p + '")' + ' ~ ' + (' + '.join(['Q("' + c + '")' for c in conf['Controls']])) for p in conf['Predictions']]
    
    #Run Regressions
    results = {}
    results_index = ['Intercept'] + conf['Controls'] + ['N', 'AIC']
    for formula, p in zip(formulas, conf['Predictions']):
        y, X = patsy.dmatrices(formula, data, return_type='dataframe')
        model = OLS(y, X).fit(maxiter=10000)
        result = {param:(coef,pvalue) for param, coef, pvalue in zip(model.params.index, model.params, model.pvalues)}
        result['N'] = int(model.nobs)
        result['AIC'] = round(model.aic, 2)
        results[p.split('_')[0]] = result
    results = pd.DataFrame(results)
    results.index = results_index
    results[:-2] = results[:-2].map(format_pvalue)
    
    #Show Results
    print(results.to_latex()) if to_latex else display(results)
    results.to_clipboard()

if __name__ == '__main__':
    #Hyperparameters
    config = [2]
    interviews = prepare_data()
    model = 'nli_sum_quant'
    waves = ['Wave 1', 'Wave 3']
    interviews = interviews[interviews[[wave + ':Interview Code' for wave in waves]].notna().all(axis=1)]

    for c in config:
        if c == 1:
            plot_ecdf(interviews, model)
        elif c == 2:
            to_latex = False            
            plot_morality_distinction(interviews, model, waves, to_latex)
        elif c == 3:
            compute_decisiveness(interviews, model, waves)
        elif c == 4:
            plot_morality_evolution(interviews, model, waves)
        elif c == 5:
            plot_morality_shift(interviews, model, waves)
        elif c == 6:
            conf = {
                    'Predictions': MORALITY_ORIGIN,
                    'Controls': ['Race', 'Gender', 'Parent Education', 'Household Income'],
                    'References': {'Attribute Names': ['Race', 'Gender', 'Parent Education'], 'Attribute Values': ['White', 'Male', '< College']}}
            to_latex = False
            predict_shift(interviews, model, conf, to_latex)