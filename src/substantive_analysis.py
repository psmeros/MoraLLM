import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
from statsmodels.discrete.discrete_model import Probit
from statsmodels.regression.linear_model import OLS
from IPython.display import display
from scipy.stats import pearsonr
from sklearn.preprocessing import scale

from __init__ import *
from src.helpers import CODED_WAVES, DEMOGRAPHICS, INCOME_RANGE, MORALITY_ORIGIN, format_pvalue
from src.parser import prepare_data

#Compute overall morality distribution
def compute_distribution(interviews, model):
    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN) for wave in CODED_WAVES]).reset_index(drop=True)
    data = data.melt(value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    data['Value'] = data['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.displot(data=data, x='Value', y='Morality', hue='Morality', hue_order=MORALITY_ORIGIN, kind='hist', legend=False, aspect=2, palette='Set2')
    g.figure.suptitle('Overall Distribution', y= 1.05, x=.5)
    g.set_ylabels('')
    g.set_xlabels('')
    g.set(xlim=(0, 100))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_distro.png', bbox_inches='tight')
    plt.show()

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

    #Compute morality shifts across waves
    shifts = []
    if len(waves) == 3:
        slice = data[data[[wave + ':Interview Code' for wave in ['Wave 1', 'Wave 2', 'Wave 3']]].notna().all(axis=1)]
        for from_wave, to_wave in zip(['Wave 1', 'Wave 2'], ['Wave 2', 'Wave 3']):
            shift = pd.DataFrame(slice[[to_wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values - slice[[from_wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
            shift[DEMOGRAPHICS] = slice[[from_wave + ':' + d for d in DEMOGRAPHICS]].values
            shift ['Count'] = .5
            shift = shift.melt(id_vars=DEMOGRAPHICS+['Count'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
            shifts.append(shift)

        slice = data[data[[wave + ':Interview Code' for wave in ['Wave 1', 'Wave 2', 'Wave 3']]].isna().any(axis=1)]
        for from_wave, to_wave in zip(['Wave 1', 'Wave 1', 'Wave 2'], ['Wave 2', 'Wave 3', 'Wave 3']):
            shift = pd.DataFrame(slice[[to_wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values - slice[[from_wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
            shift[DEMOGRAPHICS] = slice[[from_wave + ':' + d for d in DEMOGRAPHICS]].values
            shift ['Count'] = 1
            shift = shift.dropna().melt(id_vars=DEMOGRAPHICS+['Count'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
            shifts.append(shift)
    elif len(waves) == 2:
        shift = pd.DataFrame(data[[waves[1] + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values - data[[waves[0] + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
        shift[DEMOGRAPHICS] = data[[waves[0] + ':' + d for d in DEMOGRAPHICS]].values
        shift ['Count'] = 1
        shift = shift.dropna().melt(id_vars=DEMOGRAPHICS+['Count'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
        shifts.append(shift)


    #Prepare data
    shifts = pd.concat(shifts).reset_index(drop=True)
    shifts['Value'] = shifts['Value'] * 100
    shifts['Race'] = shifts['Race'].map(lambda r: {'White': 'White', 'Black': 'Other', 'Other': 'Other'}.get(r, None))
    shifts['Household Income'] = shifts['Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    for demographic in DEMOGRAPHICS:
        shifts[demographic] = shifts[demographic].map(lambda x: x + ' (N = ' + str(int(shifts[shifts[demographic] == x]['Count'].sum()/len(MORALITY_ORIGIN))) + ')')

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

#Plot Intuitive-Consequentialist and Social-Theistic Morality Distinction
def plot_morality_distinction(interviews, model, waves):

    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN) for wave in waves]).reset_index(drop=True)
    data = data * 100

    #Compute Distinction
    data = data.apply(lambda x: pd.Series([abs(x['Intuitive'] - x['Consequentialist']), abs(x['Social'] - x['Theistic'])]), axis=1)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(data=data, x=0, y=1, kind='hex', color='rosybrown')
    g.figure.suptitle('Morality Distinction', y=1.03)
    ax = plt.gca()
    ax.xaxis.set_ticks([0, 50, 100])
    ax.yaxis.set_ticks([0, 50, 100])
    ax.set_xlabel('Intuitive-Consequentialist Distinction')
    ax.set_ylabel('Social-Theistic Distinction')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
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


#Predict Survey and Oral Behavior based on Morality Origin
def compute_behavioral_regressions(interviews, confs, to_latex):
    for raw_conf in confs:
        print(raw_conf['Descrition'])

        #Run regressions with and without controls
        if raw_conf['Controls']:
            conf = raw_conf.copy()
            conf['Controls'] = []
            conf['References'] = {'Attribute Names': [], 'Attribute Values': []}
            extended_confs = [conf, raw_conf.copy()]
        else:
            extended_confs = [raw_conf]
        
        extended_results = []
        for conf in extended_confs:
            #Add Reference for Moral Schemas
            if conf['Predictors'] == ['Moral Schemas']:
                conf['References']['Attribute Names'].append('Moral Schemas')
                conf['References']['Attribute Values'].append('Theistic')
            
            #Prepare Data
            data = interviews.copy()
            data = pd.concat([pd.DataFrame(data[['Survey Id'] + [from_wave + ':' + pr for pr in conf['Predictors']] + [from_wave + ':' + c for c in conf['Controls']] + ([from_wave + ':' + p for p in conf['Predictions']] if conf['Previous Behavior'] else []) + [to_wave + ':' + p for p in conf['Predictions']]].values) for from_wave, to_wave in zip(conf['From_Wave'], conf['To_Wave'])])
            data.columns = ['Survey Id'] + conf['Predictors'] + conf['Controls'] + (conf['Predictions'] if conf['Previous Behavior'] else []) + [p + '_pred' for p in conf['Predictions']]
            data['Wave'] = pd.concat([pd.Series([float(from_wave.split()[1])] * int(len(data)/len(conf['From_Wave']))) for from_wave in conf['From_Wave']])
            data = data.map(lambda x: np.nan if x == None else x)
            data = data[~data[conf['Predictors']].isna().all(axis=1)]
            
            #Binary Representation for Probit Model
            if conf['Model']  == 'Probit':
                data[(conf['Predictions'] if conf['Previous Behavior'] else []) + [p + '_pred' for p in conf['Predictions']]] = data[(conf['Predictions'] if conf['Previous Behavior'] else []) + [p + '_pred' for p in conf['Predictions']]].map(lambda p: int(p > .5) if not pd.isna(p) else pd.NA)

            #Add Reference Controls
            for attribute_name in conf['References']['Attribute Names']:
                dummies = pd.get_dummies(data[attribute_name], prefix=attribute_name, prefix_sep=' = ').astype(int)                
                data = pd.concat([data, dummies], axis=1).drop(attribute_name, axis=1)
                c = 'Controls' if attribute_name in conf['Controls'] else 'Predictors' if attribute_name in conf['Predictors'] else None
                conf[c] = conf[c][:conf[c].index(attribute_name)] + list(dummies.columns) + conf[c][conf[c].index(attribute_name) + 1:]

            #Convert Data to Numeric
            data = data.apply(pd.to_numeric)

            #Compute Descriptive Statistics for Controls
            if conf['Controls']:
                stats = []
                for wave in conf['From_Wave']:
                    slice = data[data['Wave'] == int(wave.split()[1])]
                    stat = slice[conf['Controls']].describe(include = 'all').T[['count', 'mean', 'std', 'min', 'max']]
                    stat[['count', 'mean', 'std', 'min', 'max']] = stat.apply(lambda s: pd.Series([s.iloc[0], round(s.iloc[1], 2), round(s.iloc[2], 2), s.iloc[3], s.iloc[4]]), axis=1).astype(pd.Series([int, float, float, int, int]))
                    stats.append(stat)
                stats = pd.concat(stats, axis=1)
                stats.columns = pd.MultiIndex.from_tuples([(wave, stat) for wave in conf['From_Wave'] for stat in stats.columns[:5]])
                print(stats.to_latex()) if to_latex else display(stats)
                stats.to_clipboard()
            
            #Compute Descriptive Statistics for Predictions
            if conf['Predictions']:
                stats = []
                for wave in conf['From_Wave']:
                    slice = data[data['Wave'] == int(wave.split()[1])]
                    stat = slice[[p + '_pred' for p in conf['Predictions']]].describe(include = 'all').T[['count', 'mean', 'std', 'min', 'max']]
                    stat = stat.map(lambda x: x if not pd.isna(x) else -1)
                    stat[['count', 'mean', 'std', 'min', 'max']] = stat.apply(lambda s: pd.Series([s.iloc[0], round(s.iloc[1], 2), round(s.iloc[2], 2), s.iloc[3], s.iloc[4]]), axis=1).astype(pd.Series([int, float, float, int, int]))
                    stat = stat.map(lambda x: x if not x == -1 else '-')
                    stat['<NA>'] = slice[[p + '_pred' for p in conf['Predictions']]].isnull().sum()
                    stats.append(stat)
                stats = pd.concat(stats, axis=1)
                stats.index = conf['Predictions']
                stats.columns = pd.MultiIndex.from_tuples([(wave, stat) for wave in conf['To_Wave'] for stat in stats.columns[:6]])
                print(stats.to_latex()) if to_latex else display(stats)
                stats.to_clipboard()

            #Drop NA and Reference Dummies
            conf['Controls'] = [c for c in conf['Controls'] if c not in [attribute_name + ' = ' + attribute_value for attribute_name, attribute_value in zip(conf['References']['Attribute Names'], conf['References']['Attribute Values'])]]
            conf['Predictors'] = [c for c in conf['Predictors'] if c not in [attribute_name + ' = ' + attribute_value for attribute_name, attribute_value in zip(conf['References']['Attribute Names'], conf['References']['Attribute Values'])]]
            data = data.drop([attribute_name + ' = ' + attribute_value for attribute_name, attribute_value in zip(conf['References']['Attribute Names'], conf['References']['Attribute Values'])], axis=1)
            data = data.reset_index(drop=True)

            #Compute Results
            if conf['Model'] in ['Probit', 'OLS']:
                #Define Formulas
                formulas = ['Q("' + p + '_pred")' + ' ~ ' + ' + '.join(['Q("' + pr + '")' for pr in conf['Predictors']]) + (' + ' + ' + '.join(['Q("' + c + '")' for c in conf['Controls']]) if conf['Controls'] else '') + ('+ Q("' + p + '")' if conf['Previous Behavior'] else '') + ' + Q("Survey Id")' + (' + Q("Wave")' if conf['Dummy'] and (p not in ['Cheat', 'Cutclass', 'Secret']) else '') + (' - 1' if not conf['Intercept'] else '') for p in conf['Predictions']]
                
                #Run Regressions
                results = {}
                results_index = (['Intercept'] if conf['Intercept'] else []) + [pr.split('_')[0] for pr in conf['Predictors']] + conf['Controls'] + (['Wave'] if conf['Dummy'] else []) + (['Previous Behavior'] if conf['Previous Behavior'] else []) + ['N', 'AIC']
                for formula, p in zip(formulas, conf['Predictions']):
                    y, X = patsy.dmatrices(formula, data, return_type='dataframe')
                    groups = X['Q("Survey Id")']
                    X = X.drop('Q("Survey Id")', axis=1)
                    model = Probit if conf['Model'] == 'Probit' else OLS if conf['Model'] == 'OLS' else None
                    fit_params = {'method':'bfgs', 'disp':False} if conf['Model'] == 'Probit' else {'cov':'cluster', 'cov_kwds':{'groups': groups}} if conf['Model'] == 'OLS' else {}
                    model = model(y, X).fit(maxiter=10000, **fit_params)
                    result = {param:(coef,pvalue) for param, coef, pvalue in zip(model.params.index, model.params, model.pvalues)}
                    if conf['Previous Behavior']:
                        result['Previous Behavior'] = result['Q("' + p + '")']
                        result.pop('Q("' + p + '")')
                    result['N'] = int(model.nobs)
                    result['AIC'] = round(model.aic, 2)
                    results[p.split('_')[0]] = result
                results = pd.DataFrame(results)
                results.index = results_index

                #Scale Results
                results = pd.concat([pd.DataFrame(('(' + pd.DataFrame(scale(results[:-2].map(lambda c: c[0] if not pd.isna(c) else None))).map(str) + ',' + pd.DataFrame(results[:-2].map(lambda c: c[1] if not pd.isna(c) else None)).map(str).values + ')').values, index=results[:-2].index, columns=results[:-2].columns).map(str).replace('(nan,nan)', 'None').map(eval).map(format_pvalue), pd.DataFrame(results[-2:])])
            
            #Compute Results
            elif conf['Model'] in ['Pearson']:
                #Compute Correlations
                results = pd.DataFrame(index=[mo1 + ' - ' + mo2 for i, mo1 in enumerate(MORALITY_ORIGIN) for j, mo2 in enumerate(MORALITY_ORIGIN) if i < j] + ['N'], columns=list(set([c.split('_')[1] for c in conf['Predictors']])))
                for estimator in list(set([c.split('_')[1] for c in conf['Predictors']])):
                    slice = data[[mo + '_' + estimator + '_bin' for mo in MORALITY_ORIGIN]].dropna().reset_index(drop=True)
                    for i in results.index[:-1]:
                        results.loc[i, estimator] = format_pvalue(pearsonr(slice[i.split(' - ')[0] + '_' + estimator + '_bin'], slice[i.split(' - ')[1] + '_' + estimator + '_bin']))
                    results.loc['N', estimator] = len(slice)

            extended_results.append(results)
        
        #Concatenate Results with and without controls
        results = pd.concat(extended_results, axis=1).fillna('-')
        results = results[[pr.split('_')[0] for pr in conf['Predictions']] if conf['Predictions'] else results.columns]
        if conf['Model'] == 'Probit':
            results = pd.concat([results.drop(index=['N', 'AIC']), results.loc[['N', 'AIC']]])
        print(results.to_latex()) if to_latex else display(results)
        # results.to_clipboard()

if __name__ == '__main__':
    #Hyperparameters
    config = [7]
    extend_dataset = True
    to_latex = False
    model = 'nli_sum_quant'
    interviews = prepare_data([model], extend_dataset)

    for c in config:
        if c == 1:
            compute_distribution(interviews, model)
        elif c == 2:
            plot_ecdf(interviews, model)
        elif c == 3:
            waves = ['Wave 1', 'Wave 3']
            plot_morality_evolution(interviews, model, waves)
        elif c == 4:
            waves = ['Wave 1', 'Wave 3']
            plot_morality_shift(interviews, model, waves)
        elif c == 5:
            waves = ['Wave 1', 'Wave 3']
            plot_morality_distinction(interviews, model, waves)
        elif c == 6:
            confs = [
                        #Predicting Future Behavior [0]
                         {'Descrition': 'Predicting Future Behavior: ' + model,
                          'From_Wave': ['Wave 1', 'Wave 2', 'Wave 3'],
                          'To_Wave': ['Wave 2', 'Wave 3', 'Wave 4'],
                          'Predictors': [mo + '_' + model for mo in MORALITY_ORIGIN] if model != 'Moral Schemas' else ['Moral Schemas'],
                          'Predictions': ['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help'],
                          'Dummy' : True,
                          'Intercept': True,
                          'Previous Behavior': True,
                          'Model': 'Probit',
                          'Controls': ['Number of friends', 'Regular volunteers', 'Use drugs', 'Similar beliefs', 'Religion', 'Race', 'Gender', 'Region', 'Parent Education', 'Household Income', 'GPA'],
                          'References': {'Attribute Names': ['Religion', 'Race', 'Gender', 'Region', 'Parent Education'], 'Attribute Values': ['Catholic', 'White', 'Male', 'Not South', 'College or More']}}
                    ] + [
                        #Computing Pairwise Correlations [1]
                         {'Descrition': 'Computing Pairwise Correlations: ' + model,
                          'From_Wave': ['Wave 1', 'Wave 2', 'Wave 3'], 
                          'To_Wave': ['Wave 1', 'Wave 2', 'Wave 3'],
                          'Predictors': [mo + '_' + model for mo in MORALITY_ORIGIN],
                          'Predictions': [],
                          'Previous Behavior': False,
                          'Model': 'Pearson',
                          'Controls': [],
                          'References': {'Attribute Names': [], 'Attribute Values': []}}
                    ] + [
                        #Estimating Morality Sources from Social Categories (OLS) [2]
                         {'Descrition': 'Estimating Morality Sources from Social Categories (OLS): ' + model,
                          'From_Wave': ['Wave 1', 'Wave 2', 'Wave 3'], 
                          'To_Wave': ['Wave 1', 'Wave 2', 'Wave 3'],
                          'Predictors': ['Verbosity', 'Uncertainty', 'Complexity', 'Sentiment'],
                          'Predictions': [mo + '_' + model for mo in MORALITY_ORIGIN],
                          'Dummy' : True,
                          'Intercept': True,
                          'Previous Behavior': False,
                          'Model': 'OLS',
                          'Controls': ['Number of friends', 'Regular volunteers', 'Use drugs', 'Similar beliefs', 'Religion', 'Race', 'Gender', 'Region', 'Parent Education', 'Household Income', 'GPA'],
                          'References': {'Attribute Names': ['Religion', 'Race', 'Gender', 'Region', 'Parent Education'], 'Attribute Values': ['Catholic', 'White', 'Male', 'Not South', 'College or More']}}
                    ]
            confs = [confs[2]]
            compute_behavioral_regressions(interviews, confs, to_latex)
        elif c == 7:
            compute_decisiveness(interviews, model, ['Wave 1', 'Wave 3'])