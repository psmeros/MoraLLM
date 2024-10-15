import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display
from scipy.stats import fisher_exact, pearsonr
from sklearn.preprocessing import minmax_scale, normalize, scale

from __init__ import *
from src.helpers import ADOLESCENCE_RANGE, CHURCH_ATTENDANCE_RANGE, CODED_WAVES, DEMOGRAPHICS, EDUCATION_RANGE, INCOME_RANGE, MORAL_SCHEMAS, MORALITY_ESTIMATORS, MORALITY_ORIGIN, format_pvalue
from src.parser import prepare_data

#Compute overall morality distribution
def compute_distribution(interviews):
    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN) for wave in CODED_WAVES]).reset_index(drop=True)
    data = data.melt(value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    data['Value'] = data['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=data, x='Value', y='Morality', hue='Morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='boxen', width=.7, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Overall Distribution', y= 1.05, x=.5)
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_distro.png', bbox_inches='tight')
    plt.show()

#Plot morality shifts
def plot_morality_shifts(interviews, attributes, shift_threshold):

    #Compute morality shifts across waves
    def compute_morality_shifts(interviews, attribute_name=None, attribute_value=None):
        #Prepare data 
        if attribute_name is not None:
            interviews = interviews[interviews[CODED_WAVES[0] + ':' + attribute_name] == attribute_value]
        N = len(interviews)

        wave_source = interviews[[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]]
        wave_target = interviews[[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]]
        wave_source.columns = MORALITY_ORIGIN
        wave_target.columns = MORALITY_ORIGIN

        #Compute normalized shift
        outgoing = (wave_source - wave_target).clip(lower=0)
        incoming = pd.DataFrame(normalize((wave_target - wave_source).clip(lower=0), norm='l1'))

        #Compute shifts
        shifts = []
        for i in range(N):
            shift = pd.DataFrame(outgoing.iloc[i]).values.reshape(-1, 1) @ pd.DataFrame(incoming.iloc[i]).values.reshape(1, -1)
            shift = pd.DataFrame(shift, index=[CODED_WAVES[0] + ':' + mo for mo in MORALITY_ORIGIN], columns=[CODED_WAVES[1] + ':' + mo for mo in MORALITY_ORIGIN])
            shift = shift.stack().reset_index().rename(columns={'level_0':'source', 'level_1':'target', 0:'value'})

            shift['wave'] = shift.apply(lambda x: x['source'].split(':')[0] + '->' + x['target'].split(':')[0].split()[1], axis=1)
            shift['source'] = shift['source'].apply(lambda x: x.split(':')[-1])
            shift['target'] = shift['target'].apply(lambda x: x.split(':')[-1])
            source_shift = shift.drop('target', axis=1).rename(columns={'source':'morality'})
            source_shift['value'] = -source_shift['value']
            target_shift = shift.drop('source', axis=1).rename(columns={'target':'morality'})
            shift = pd.concat([source_shift, target_shift])
            shift = shift[abs(shift['value']) > shift_threshold]
            shift['value'] = shift['value'] * 100

            shifts.append(shift)
        shifts = pd.concat(shifts)

        return shifts, N

    #Prepare data
    data = interviews.copy()
    data = pd.DataFrame(data[[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values - data[[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    data = data.melt(value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    data['Value'] = data['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=data, x='Value', y='Morality', hue='Morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Crosswave Morality Development', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-10, 10))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_diff_distro.png', bbox_inches='tight')
    plt.show()

    #Prepare data
    data = interviews.copy()
    data[CODED_WAVES[0] + ':Adolescence'] = data[CODED_WAVES[0] + ':Age'].map(lambda x: ADOLESCENCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Household Income'] = data[CODED_WAVES[0] + ':Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].map(lambda x: CHURCH_ATTENDANCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Parent Education'] = data[CODED_WAVES[0] + ':Parent Education'].map(lambda x: EDUCATION_RANGE.get(x, None))
    data = data.dropna(subset=[wave + ':Interview Code' for wave in CODED_WAVES])

    shifts, _ = compute_morality_shifts(data)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=shifts, x='value', y='morality', hue='morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Crosswave Morality Diffusion', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-10, 10))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_shift.png', bbox_inches='tight')
    plt.show()

    #Prepare data
    data[[wave + ':Race' for wave in CODED_WAVES]] = data[[wave + ':Race' for wave in CODED_WAVES]].map(lambda r: {'White': 'White', 'Black': 'Other', 'Other': 'Other'}.get(r, None))
    shifts = []
    legends = {}
    symbols = ['■ ', '▼ ']

    for attribute in attributes:
        legend = []
        for attribute_value in attribute['values']:
            shift, N = compute_morality_shifts(data, attribute_name=attribute['name'], attribute_value=attribute_value)
            if not shift.empty:
                shift['Attribute'] = attribute['name']
                legend.append(symbols[attribute['values'].index(attribute_value)] + attribute_value + ' (N = ' + str(N) + ')')
                shift['order'] = str(attribute['values'].index(attribute_value)) + shift['morality'].apply(lambda mo: str(MORALITY_ORIGIN.index(mo)))
                shifts.append(shift)
        legends[attribute['name']] = attribute['name'] + '\n' + ', '.join(legend) + '\n'

    shifts = pd.concat(shifts)
    shifts = shifts.sort_values(by='order')
    shifts['Attribute'] = shifts['Attribute'].map(legends)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=shifts, x='value', y='morality', hue='order', orient='h', order=MORALITY_ORIGIN, col='Attribute', col_order=legends.values(), col_wrap=3, kind='point', err_kws={'linewidth': 3}, dodge=.7, markers=['s']*len(MORALITY_ORIGIN)+['v']*len(MORALITY_ORIGIN), markersize=15, legend=False, seed=42, aspect=1.5, palette=2*sns.color_palette('Set2', n_colors=len(MORALITY_ORIGIN)))
    g.figure.suptitle('Crosswave Morality Diffusion by Social Categories', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-10, 10))
    g.set_xlabels('')
    g.set_ylabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    g.set_titles('{col_name}')
    plt.subplots_adjust(wspace=.3)
    plt.savefig('data/plots/fig-morality_shift_by_attribute.png', bbox_inches='tight')
    plt.show()

#Compute crosswave consistency
def compute_consistency(interviews, plot_type, consistency_threshold):
    data = interviews.copy()
    data = data.dropna(subset=[wave + ':Interview Code' for wave in CODED_WAVES])
    
    #Prepare Data
    data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]] = minmax_scale(data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]])
    consistency = data.apply(lambda i: pd.Series(abs(i[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] - (i[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]]) < consistency_threshold) for mo in MORALITY_ORIGIN), axis=1).set_axis([mo for mo in MORALITY_ORIGIN], axis=1)
    consistency = (consistency.mean()) * 100
    consistency = consistency.reset_index()
    consistency.columns = ['morality', 'r']
    consistency['morality-r'] = consistency['morality'] + consistency['r'].apply(lambda r: ' (' + str(round(r, 1))) + '%)'
    consistency['angles'] = np.linspace(0, 2 * np.pi, len(consistency), endpoint=False)
    consistency.loc[len(consistency)] = consistency.iloc[0]

    #Plot
    if plot_type == 'spider':
        plt.figure(figsize=(10, 10))
        _, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(consistency['angles'], consistency['r'], linewidth=2, linestyle='solid', color='rosybrown', alpha=0.8)
        ax.fill(consistency['angles'], consistency['r'], 'rosybrown', alpha=0.7)
        ax.set_theta_offset(np.pi)
        ax.set_theta_direction(-1)
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_xticks(consistency['angles'], [])
        num_levels = 4
        for i in range(1, num_levels + 1):
            level =  100 * i / num_levels
            level_values = [level] * len(consistency)
            ax.plot(consistency['angles'], level_values, color='gray', linestyle='--', linewidth=0.7)
        for i in range(len(consistency)):
            ax.plot([consistency['angles'].iloc[i], consistency['angles'].iloc[i]], [0, 100], color='gray', linestyle='-', linewidth=0.7)
        for i, (r, horizontalalignment, verticalalignment, rotation) in enumerate(zip([105, 115, 105, 115], ['right', 'center', 'left', 'center'], ['center', 'top', 'center', 'bottom'], [90, 0, -90, 0])):
            ax.text(consistency['angles'].iloc[i], r, consistency['morality-r'].iloc[i], size=15, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation=rotation)
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], [])
        plt.title('Crosswave Interviewees Consistency', y=1.15, size=20)
        plt.savefig('data/plots/fig-morality_consistency.png', bbox_inches='tight')
        plt.show()
    
    elif plot_type == 'bar':
        sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
        plt.figure(figsize=(20, 10))
        g = sns.catplot(data=consistency, x='r', y='morality', hue='morality', orient='h', order=MORALITY_ORIGIN, kind='bar', seed=42, aspect=2, legend=False, palette=sns.color_palette('Set2')[:4])
        g.set_xlabels('')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.title('Crosswave Interviewees Consistency')
        plt.savefig('data/plots/fig-morality_consistency.png', bbox_inches='tight')
        plt.show()

#Plot Intuitive-Consequentialist and Social-Theistic Morality Distinction
def plot_morality_distinction(interviews):

    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN) for wave in CODED_WAVES]).reset_index(drop=True)

    #Normalize Data
    data = pd.DataFrame(minmax_scale(data), columns=MORALITY_ORIGIN)
    data = data * 100

    #Compute Distinction
    data = data.apply(lambda x: pd.Series([abs(x['Intuitive'] - x['Consequentialist']), abs(x['Social'] - x['Theistic'])]), axis=1)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(data=data, x=0, y=1, kind='hex', color='rosybrown')
    g.figure.suptitle('Morality Sources Distinction', y=1.03)
    ax = plt.gca()
    ax.xaxis.set_ticks([0, 50, 100])
    ax.yaxis.set_ticks([0, 50, 100])
    ax.set_xlabel('Intuitive-Consequentialist Distinction')
    ax.set_ylabel('Social-Theistic Distinction')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_distinction.png', bbox_inches='tight')
    plt.show()

#Compute Morality and Behavioral Correlations
def compute_morality_correlations(interviews, correlation_type, to_latex):
    compute_pearsonr = lambda x, y: pearsonr(x, y)
    compute_fisher = lambda x, y: (lambda x, y: fisher_exact([[np.sum(x & y), np.sum(~x & y)], [np.sum(x & ~y), np.sum(~x & ~y)]]))(x.round().astype(bool), y.round().astype(bool))
    compute_rlm = lambda x, y: (lambda lm: (lm.params['x'],lm.pvalues['x']))(smf.rlm(formula='y ~ x', data=pd.concat([x, y], axis=1).rename(columns={0:'x', 1:'y'}), M=sm.robust.norms.AndrewWave()).fit())
    compute_correlation = lambda x, y: format_pvalue(compute_pearsonr(x, y)) if correlation_type == 'pearsonr' else format_pvalue(compute_rlm(x, y)) if correlation_type == 'rlm' else format_pvalue(compute_fisher(x, y)) if correlation_type == 'fisher' else None

    #Prepare Data
    data = interviews.copy()
    data = pd.concat([data[[wave + ':' + mo + '_' + estimator for mo in MORALITY_ORIGIN for estimator in MORALITY_ESTIMATORS]].rename(columns = {wave + ':' + mo + '_' + estimator : mo + '_' + estimator for mo in MORALITY_ORIGIN for estimator in MORALITY_ESTIMATORS}) for wave in CODED_WAVES])
    data = data.dropna()
    data = data.apply(pd.to_numeric)
    
    #Compute Morality Correlations
    correlations = pd.DataFrame(index=MORALITY_ORIGIN, columns=MORALITY_ORIGIN)
    for i, mo1 in enumerate(MORALITY_ORIGIN):
        for j, mo2 in enumerate(MORALITY_ORIGIN):
            if i != j:
                correlations.loc[mo1, mo2] = compute_correlation(data[mo1 + '_' + MORALITY_ESTIMATORS[i<j]], data[mo2 + '_' + MORALITY_ESTIMATORS[i<j]])

    correlations = correlations.astype(str).replace('nan', '')
    print(correlations.to_latex()) if to_latex else display(correlations)
    print('N =', len(data))

    #Prepare Data
    data = interviews.copy()
    data = pd.concat([data[[wave + ':' + mo + '_' + estimator for mo in MORALITY_ORIGIN for estimator in MORALITY_ESTIMATORS] + [wave + ':Moral Schemas']].rename(columns = {wave + ':' + mo + '_' + estimator : mo + '_' + estimator for mo in MORALITY_ORIGIN for estimator in MORALITY_ESTIMATORS}).rename(columns = {wave + ':Moral Schemas' : 'Moral Schemas'}) for wave in CODED_WAVES])
    data = data.dropna()
    data = pd.concat([data, pd.get_dummies(data['Moral Schemas']).astype(float)], axis=1).drop('Moral Schemas', axis=1)
    data = data.apply(pd.to_numeric)

    #Compute Behavioral Correlations
    correlations = pd.DataFrame(index=MORAL_SCHEMAS.values(), columns=pd.MultiIndex.from_tuples([(estimator, mo) for estimator in MORALITY_ESTIMATORS for mo in MORALITY_ORIGIN]))
    for estimator in MORALITY_ESTIMATORS:
        for i, ms in enumerate(MORAL_SCHEMAS.values()):
            for j, mo in enumerate(MORALITY_ORIGIN):
                correlations.loc[ms, (estimator, mo)] = compute_correlation(data[mo + '_' + estimator], data[ms])

    correlations = correlations.astype(str).replace('nan', '')
    print(correlations.to_latex()) if to_latex else display(correlations)
    print('N =', len(data))

#Predict Morality Origin based on Linguistic Attributes
def compute_linguistic_regressions(interviews, linguistic_attributes, to_latex):
    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(interviews[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN] + [wave + ':' + la for la in linguistic_attributes]].values) for wave in CODED_WAVES])
    data.columns = MORALITY_ORIGIN + linguistic_attributes
    data = data.dropna().apply(pd.to_numeric)
    data = pd.DataFrame(scale(data), columns=data.columns)

    formulas = [mo + ' ~ ' + ' + '.join(linguistic_attributes) + ' - 1' for mo in MORALITY_ORIGIN]

    #Display Results
    results = {}
    for formula, mo in zip(formulas, MORALITY_ORIGIN):
        lm = smf.rlm(formula=formula, data=data, M=sm.robust.norms.AndrewWave()).fit()
        result = {param:format_pvalue((coef,pvalue)) for param, coef, pvalue in zip(lm.params.index, lm.params, lm.pvalues)}
        results[mo] = result
        
    results = pd.DataFrame(results)
    print(results.to_latex()) if to_latex else display(results)
    print('N =', len(data))

#Predict Behavioral Actions based on Morality Origin
def compute_behavioral_regressions(interviews, behaviors, to_latex):
    for behavior in behaviors:
        #Prepare Data
        data = interviews.copy()
        data = pd.concat([data, pd.get_dummies(data[[wave + ':Moral Schemas' for wave in CODED_WAVES]]).rename(columns = {wave + ':Moral Schemas_' + ms : wave + ':' + ms + '_Moral Schemas' for wave in CODED_WAVES for ms in MORAL_SCHEMAS.values()}).astype(float)], axis=1).drop([wave + ':Moral Schemas' for wave in CODED_WAVES], axis=1)
        data[[wave + ':' + action for wave in ['Wave 3', 'Wave 4'] for action in ['Cheat', 'Cutclass', 'Secret']]] = pd.NA
        data = pd.concat([pd.DataFrame(data[[from_wave + ':' + pr for pr in behavior['Predictors']] + [from_wave + ':' + c for c in behavior['Controls']] + [from_wave + ':' + a for a in behavior['Actions']] + [to_wave + ':' + a for a in behavior['Actions']]].values) for from_wave, to_wave in zip(behavior['From_Wave'], behavior['To_Wave'])])
        data.columns = behavior['Predictors'] + behavior['Controls'] + behavior['Actions'] + [a + '_pred' for a in behavior['Actions']]
        data = data.dropna(subset = behavior['Predictors'] + behavior['Controls'])
        
        for attribute_name, attribute_value in zip(behavior['References']['Attribute Names'], behavior['References']['Attribute Values']):
            dummies = pd.get_dummies(data[attribute_name], prefix=attribute_name, prefix_sep=' = ').drop(attribute_name + ' = ' + attribute_value, axis=1).astype(float)
            data = pd.concat([data, dummies], axis=1)
            data = data.drop(attribute_name, axis=1)
            behavior['Controls'] = behavior['Controls'][:behavior['Controls'].index(attribute_name)] + list(dummies.columns) + behavior['Controls'][behavior['Controls'].index(attribute_name) + 1:]
        data = data.apply(pd.to_numeric)

        #Display Results
        formulas = [a + '_pred' + ' ~ ' + ' + '.join(['Q("' + pr + '")' for pr in behavior['Predictors']]) + (' + ' + ' + '.join(['Q("' + c + '")' for c in behavior['Controls'] + [a]]) if behavior['Controls'] else '') + ' - 1' for a in behavior['Actions']]
        results = {}
        for formula, a in zip(formulas, behavior['Actions']):
            probit = smf.probit(formula=formula, data=data).fit(disp=False, cov_type='HC3')
            result = {param:format_pvalue((coef,pvalue)) for param, coef, pvalue in zip(probit.params.index, probit.params, probit.pvalues)}
            result['Previous Behavior'] = result['Q("' + a + '")']
            result.pop('Q("' + a + '")')
            results[a + ' (N = ' + str(probit.nobs) + ')'] = result
            
        results = pd.DataFrame(results)
        results.index = [pr.split('_')[0] for pr in behavior['Predictors']] + behavior['Controls'] + ['Previous Behavior']
        print(results.to_latex()) if to_latex else display(results)

if __name__ == '__main__':
    #Hyperparameters
    config = [7]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    extend_dataset = True
    to_latex = False
    interviews = prepare_data(interviews, extend_dataset)

    for c in config:
        if c == 1:
            compute_distribution(interviews)
        elif c == 2:
            shift_threshold = 0
            attributes = DEMOGRAPHICS
            plot_morality_shifts(interviews, attributes, shift_threshold)
        elif c == 3:
            consistency_threshold = .1
            plot_type = 'spider'
            compute_consistency(interviews, plot_type, consistency_threshold)
        elif c == 4:
            plot_morality_distinction(interviews)
        elif c == 5:
            correlation_type = 'pearsonr'
            compute_morality_correlations(interviews, correlation_type, to_latex)
        elif c == 6:
            linguistic_attributes = ['Verbosity', 'Uncertainty', 'Readability', 'Sentiment']
            compute_linguistic_regressions(interviews, linguistic_attributes, to_latex)
        elif c == 7:
            behaviors = [{'From_Wave': ['Wave 1', 'Wave 3'], 
                          'To_Wave': ['Wave 2', 'Wave 4'],
                          'Predictors': [mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN],
                          'Actions': ['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help'],
                          'Controls': ['Race', 'Gender', 'Age', 'Household Income', 'Parent Education', 'GPA', 'Church Attendance'],
                          'References': {'Attribute Names': ['Race', 'Gender'], 'Attribute Values': ['White', 'Male']}},
                         {'From_Wave': ['Wave 1', 'Wave 3'], 
                          'To_Wave': ['Wave 2', 'Wave 4'],
                          'Predictors': [ms + '_' + 'Moral Schemas' for ms in MORAL_SCHEMAS.values()],
                          'Actions': ['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help'],
                          'Controls': ['Race', 'Gender', 'Age', 'Household Income', 'Parent Education', 'GPA', 'Church Attendance'],
                          'References': {'Attribute Names': ['Race', 'Gender'], 'Attribute Values': ['White', 'Male']}}]
            compute_behavioral_regressions(interviews, behaviors, to_latex)