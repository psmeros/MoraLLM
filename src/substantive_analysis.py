import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display
from sklearn.preprocessing import minmax_scale, normalize, scale
from statsmodels.stats.diagnostic import het_white

from __init__ import *
from src.helpers import CODED_WAVES, DEMOGRAPHICS, MORALITY_ESTIMATORS, MORALITY_ORIGIN
from src.parser import merge_codings, merge_matches, merge_surveys


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

    data = interviews.copy()
    data[CODED_WAVES[0] + ':Race'] = data[CODED_WAVES[0] + ':Race'].apply(lambda x: x if x in ['White'] else 'Other')
    data[CODED_WAVES[0] + ':Adolescence'] = data[CODED_WAVES[0] + ':Age'].apply(lambda x: 'Early' if x is not pd.NA and x in ['13', '14', '15'] else 'Late' if x is not pd.NA and x in ['16', '17', '18', '19'] else '')
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].apply(lambda x: 'Irregular' if x is not pd.NA and x in [1,2,3,4] else 'Regular' if x is not pd.NA and x in [5,6] else '')

    #Prepare data
    shifts, _ = compute_morality_shifts(data)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=shifts, x='value', y='morality', hue='morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Crosswave Morality Shift', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-10, 10))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_shift.png', bbox_inches='tight')
    plt.show()

    #Prepare data
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
    g.figure.suptitle('Crosswave Shift by Social Categories', x=.5)
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

def compute_consistency(interviews, plot_type, consistency_threshold):
    data = interviews.copy()
    
    #Prepare Data
    data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]] = minmax_scale(data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]])
    consistency = interviews.apply(lambda i: pd.Series(abs(i[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] - (i[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]]) < consistency_threshold) for mo in MORALITY_ORIGIN), axis=1).set_axis([mo for mo in MORALITY_ORIGIN], axis=1)
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
        ax.set_theta_offset(np.pi)
        ax.set_theta_direction(-1)
        ax.set_xticks(consistency['angles'], [])
        for i, (r, horizontalalignment, verticalalignment, rotation) in enumerate(zip([105, 115, 105, 115], ['right', 'center', 'left', 'center'], ['center', 'top', 'center', 'bottom'], [90, 0, -90, 0])):
            ax.text(consistency['angles'].iloc[i], r, consistency['morality-r'].iloc[i], size=15, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation=rotation)
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], [])
        plt.ylim(0, 100)
        ax.plot(consistency['angles'], consistency['r'], linewidth=2, linestyle='solid', color='rosybrown', alpha=0.8)
        ax.fill(consistency['angles'], consistency['r'], 'rosybrown', alpha=0.7)
        plt.title('Crosswave Interviewees Consistency', y=1.15)
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

#Compute overall morality distribution
def compute_distribution(interviews):
    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN) for wave in CODED_WAVES]).reset_index(drop=True)
    data = data.melt(value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    data['Value'] = data['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=data, x='Value', y='Morality', hue='Morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='boxen', width=.7, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Overall Distribution', y= 1.05, x=.5)
    # g.set(xlim=(0, 60))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_distro.png', bbox_inches='tight')
    plt.show()

    data = interviews.copy()
    data = pd.DataFrame(data[[CODED_WAVES[1] + ':' + mo for mo in MORALITY_ORIGIN]].values - data[[CODED_WAVES[0] + ':' + mo for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    data = data.melt(value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    data['Value'] = data['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=data, x='Value', y='Morality', hue='Morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Crosswave Morality Change', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-10, 10))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_diff_distro.png', bbox_inches='tight')
    plt.show()

def compute_decisiveness(interviews):
    decisive_threshold = {mo + ':' + wave : np.mean(interviews[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0]]) for mo in MORALITY_ORIGIN for wave in CODED_WAVES}

    #Prepare Data
    decisiveness_options = ['Rigidly Decisive', 'Ambivalent', 'Rigidly Indecisive']
    decisiveness = interviews.apply(lambda i: pd.Series(((i[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] >= decisive_threshold[mo + ':' + CODED_WAVES[0]]), (i[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] >= decisive_threshold[mo + ':' + CODED_WAVES[1]])) for mo in MORALITY_ORIGIN), axis=1).set_axis([mo for mo in MORALITY_ORIGIN], axis=1)
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
    plt.title('Crosswave Morality Rigidity')
    plt.legend(bbox_to_anchor=(1, 1.03)).set_frame_on(False)
    plt.savefig('data/plots/fig-decisiveness.png', bbox_inches='tight')
    plt.show()

def compute_morality_correlations(interviews, model, show_plots=False):
    #Prepare Data
    data = interviews.copy()
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].apply(lambda x: 'Irregular' if x is not pd.NA and x in [1,2,3,4] else 'Regular' if x is not pd.NA and x in [5,6] else '')
    data[CODED_WAVES[1] + ':Parent Education'] = data[CODED_WAVES[0] + ':Parent Education']
    data[CODED_WAVES[1] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance']
    attribute_list = ['Morality_Origin_Word_Count', 'Morality_Origin_Uncertain_Terms', 'Morality_Origin_Readability', 'Morality_Origin_Sentiment', 'Gender', 'Race', 'Household Income', 'Parent Education', 'Age', 'Church Attendance', 'Wave']
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo for mo in MORALITY_ORIGIN + attribute_list]].values, columns=MORALITY_ORIGIN + attribute_list) for wave in CODED_WAVES]).reset_index(drop=True)

    data['Verbosity'] = minmax_scale(np.log(data['Morality_Origin_Word_Count'].astype(int)))
    data['Brevity'] = (data['Verbosity'] < 3).astype(int)
    data['Uncertainty'] = minmax_scale(data['Morality_Origin_Uncertain_Terms'].astype(int) / data['Morality_Origin_Word_Count'].astype(int))
    data['Readability'] = minmax_scale((data['Morality_Origin_Readability']).astype(float))
    data['Sentiment'] = minmax_scale(data['Morality_Origin_Sentiment'].astype(float))

    data['Gender'] = (data['Gender'] == 'Male').astype(int)
    data['Race'] = (data['Race'] == 'White').astype(int)
    data['Household_Income'] = (data['Household Income'] == 'High').astype(int)
    data['Parent_Education'] = (data['Parent Education'] == 'Tertiary').astype(int)
    data['Age'] = data['Age'].fillna(data['Age'].dropna().astype(int).mean()).astype(int)
    data['Church_Attendance'] = (data['Church Attendance'] == 'Regular').astype(int)
    data['Wave'] = (data['Wave'] == CODED_WAVES[0]).astype(int)

    data[MORALITY_ORIGIN] = scale(data[MORALITY_ORIGIN], with_mean=True, with_std=False) + .5

    data = data.melt(id_vars=['Verbosity', 'Brevity', 'Uncertainty', 'Readability', 'Sentiment', 'Gender', 'Race', 'Household_Income', 'Parent_Education', 'Age', 'Church_Attendance', 'Wave'], value_vars=MORALITY_ORIGIN, var_name='Morality Category', value_name='morality').dropna()
    data['morality'] = data['morality'].astype(float)
    formulas = ['morality ~ Verbosity',
                'morality ~ Uncertainty',
                'morality ~ Readability',
                'morality ~ Sentiment',
                'morality ~ Verbosity + Uncertainty + Readability + Sentiment',
                'morality ~ Gender',
                'morality ~ Race',
                'morality ~ Household_Income',
                'morality ~ Parent_Education',
                'morality ~ Age',
                'morality ~ Church_Attendance',
                'morality ~ Gender + Race + Household_Income + Parent_Education + Age + Church_Attendance']

    #Display Results
    compute_coef = lambda x: str(round(x[0], 2)).replace('0.', '.') + ('***' if float(x[1])<.005 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '')
    for formula in formulas:
        results = []
        for mo in MORALITY_ORIGIN:
            slice = data[data['Morality Category'] == mo]
            if model == 'ols':
                lm = smf.ols(formula=formula, data=slice).fit(cov_type='HC3')
            elif model == 'rlm':
                lm = smf.rlm(formula=formula, data=slice, M=sm.robust.norms.AndrewWave()).fit()
            results.append({param:compute_coef((coef,pvalue)) for param, coef, pvalue in zip(lm.params.index, lm.params, lm.pvalues)})
        results = pd.DataFrame(results, index=MORALITY_ORIGIN)
        display(results)

    if show_plots:
        heteroscedasticity = []
        for mo in MORALITY_ORIGIN:
            slice = data[data['Morality Category'] == mo]
            lm = smf.ols(formula=formula, data=slice).fit()
            white_test = het_white(lm.resid, sm.add_constant(slice[['Verbosity', 'Age']]))
            heteroscedasticity.append({'Heteroscedasticity':compute_coef((white_test[0], white_test[1]))})
        display(pd.DataFrame(heteroscedasticity, index=MORALITY_ORIGIN))

        data = data.melt(id_vars=['Morality Category', 'morality'], value_vars=['Verbosity', 'Age'], var_name='Attribute Name', value_name='Attribute')

        #Plot
        sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
        plt.figure(figsize=(20, 10))
        g = sns.lmplot(data=data, x='Attribute', y='morality', hue='Morality Category', col='Attribute Name', scatter=False, seed=42, facet_kws={'sharex':False}, robust=True, aspect=1.2, palette=sns.color_palette('Set2'))
        g.set_titles('{col_name}')
        g.set_ylabels('Mean-Centered Morality')
        for ax, label in zip(g.axes.flat, ['Log(Word Count)', 'Years']):
            ax.set_xlabel(label)
        g.legend.set_title('')
        plt.savefig('data/plots/fig-morality_correlations.png', bbox_inches='tight')
        plt.show()

def compute_std_diff(interviews, attributes):
    #Prepare Data
    data = interviews[[wave + ':' + mo for mo in MORALITY_ORIGIN for wave in CODED_WAVES] + [CODED_WAVES[0] + ':' + attribute['name'] for attribute in attributes]]
    data[CODED_WAVES[0] + ':Race'] = data[CODED_WAVES[0] + ':Race'].apply(lambda x: x if x in ['White'] else 'Other')
    data[CODED_WAVES[0] + ':Adolescence'] = data[CODED_WAVES[0] + ':Age'].apply(lambda x: 'Early' if x is not pd.NA and x in ['13', '14', '15'] else 'Late' if x is not pd.NA and x in ['16', '17', '18', '19'] else '')
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].apply(lambda x: 'Irregular' if x is not pd.NA and x in [1,2,3,4] else 'Regular' if x is not pd.NA and x in [5,6] else '')

    #Melt Data
    data = data.melt(id_vars=[CODED_WAVES[0] + ':' + attribute['name'] for attribute in attributes], value_vars=[wave + ':' + mo for mo in MORALITY_ORIGIN for wave in CODED_WAVES], var_name='Morality', value_name='Value')
    data['Wave'] = data['Morality'].apply(lambda x: x.split(':')[0])
    data['Morality'] = data['Morality'].apply(lambda x: x.split(':')[1].split('_')[0])
    data = data.rename(columns = {CODED_WAVES[0] + ':' + attribute['name'] : attribute['name'] for attribute in attributes})

    #Compute Standard Deviation
    stds = []
    for attribute in attributes:
        for j, attribute_value in enumerate(attribute['values']):
            slice = data[data[attribute['name']] == attribute_value]
            N = int(len(slice)/len(MORALITY_ORIGIN)/len(CODED_WAVES))
            slice = slice.groupby(['Wave', 'Morality'])['Value'].std().reset_index()
            slice = slice[slice['Wave'] == CODED_WAVES[0]][['Morality', 'Value']].merge(slice[slice['Wave'] == CODED_WAVES[1]][['Morality', 'Value']], on='Morality', suffixes=('_0', '_1'))
            std = round(((slice['Value_1'] - slice['Value_0'])/slice['Value_0']).mean() * 100, 1)
            std = {'Attribute Name' : attribute['name'], 'Attribute Position' : j, 'Attribute Value' : attribute_value + ' (N = ' + str(N) + ')', 'STD' : std}
            stds.append(std)
    stds = pd.DataFrame(stds)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=5.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=stds, x='STD', y='Attribute Position', hue='Attribute Position', col='Attribute Name', sharey=False, col_wrap=2, orient='h', kind='bar', seed=42, aspect=4, legend=False, palette=sns.color_palette('Set2')[-2:])
    g.set(xlim=(-30, 0))
    g.figure.subplots_adjust(wspace=0.55)
    g.figure.suptitle('Standard Deviation Crosswave Shift', y=1.03)
    g.set_titles('{col_name}')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    for j, ax in enumerate(g.axes):
        ax.set_ylabel('')
        labels = stds.iloc[2*j:2*j+2]['Attribute Value'].to_list()
        ax.set(yticks=range(len(labels)), yticklabels=labels)
    plt.savefig('data/plots/fig-std_diff.png', bbox_inches='tight')
    plt.show()

def print_cases(interviews, demographics_cases, incoherent_cases, max_diff_cases):
    #Prepare Data
    data = interviews.copy()
    data[CODED_WAVES[0] + ':Race'] = data[CODED_WAVES[0] + ':Race'].apply(lambda x: x if x in ['White'] else 'Other')
    data[CODED_WAVES[0] + ':Adolescence'] = data[CODED_WAVES[0] + ':Age'].apply(lambda x: 'Early' if x is not pd.NA and x in ['13', '14', '15'] else 'Late' if x is not pd.NA and x in ['16', '17', '18', '19'] else '')
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].apply(lambda x: 'Irregular' if x is not pd.NA and x in [1,2,3,4] else 'Regular' if x is not pd.NA and x in [5,6] else '')

    #Print Demographics Cases
    for case in demographics_cases:
        for c in case:
            slice = data[pd.concat([data[CODED_WAVES[0] + ':' + attribute] == c['demographics'][attribute] for attribute in c['demographics'].keys()], axis=1).all(axis=1)]
            slice['Diff'] = slice[CODED_WAVES[1] + ':' + c['morality']] - slice[CODED_WAVES[0] + ':' + c['morality']]
            slice = slice.sort_values(by='Diff', ascending=c['ascending'])
            print(slice.iloc[c['pos']][CODED_WAVES[0] + ':Morality_Origin'])
            print(slice.iloc[c['pos']][CODED_WAVES[1] + ':Morality_Origin'])
            print(str(round(slice.iloc[c['pos']][CODED_WAVES[0] + ':' + c['morality'] + '_' + MORALITY_ESTIMATORS[0]], 2) * 100) + '%' + ' → ' + str(round(slice.iloc[c['pos']][CODED_WAVES[1] + ':' + c['morality'] + '_' + MORALITY_ESTIMATORS[0]], 2) * 100) + '%' + ' (' + c['morality'] + ')')
            print('\n----------\n')
        print('\n==========\n')

    #Print Incoherent Cases
    slice = data[pd.DataFrame([data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].sum(axis=1) > len(MORALITY_ORIGIN)/2 for wave in CODED_WAVES]).T.apply(lambda w: w[0] | w[1], axis=1)]
    for ic in incoherent_cases:
        print(slice.iloc[ic][[CODED_WAVES[0] + d for d in [':Age', ':Gender', ':Race', ':Income', ':Parent Education']]].values)
        for wave in CODED_WAVES:
            print(wave)
            print(slice.iloc[ic][wave + ':Morality_Origin'])
            print(slice.iloc[ic][[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].apply(lambda x: str(int(x * 100)) + '%'))
            print(slice.iloc[ic][[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].apply(lambda x: str(int(x * 100)) + '%'))
            print('\n----------\n')

    #Print Max Diff Cases
    model_diff = pd.DataFrame(data[[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values - interviews[[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    coders_diff = pd.DataFrame(data[[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].values - interviews[[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    data['model-coders_diff'] = abs(model_diff - coders_diff).max(axis=1)
    data['model-coders_diff_morality'] = abs(model_diff - coders_diff).idxmax(axis=1)
    data = data.sort_values(by='model-coders_diff', ascending=False)

    for ic in max_diff_cases:
        print(data.iloc[ic][[CODED_WAVES[0] + d for d in [':Age', ':Gender', ':Race', ':Income', ':Parent Education']]].values)
        print(data.iloc[ic]['model-coders_diff_morality'])
        for wave in CODED_WAVES:
            print(wave)
            print(data.iloc[ic][wave + ':Morality_Origin'])
            print(data.iloc[ic][[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].apply(lambda x: str(int(x * 100)) + '%'))
            print(data.iloc[ic][[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].apply(lambda x: str(int(x * 100)) + '%'))
            print('\n----------\n')


if __name__ == '__main__':
    #Hyperparameters
    config = [2]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews)
    interviews = merge_codings(interviews)
    interviews = merge_matches(interviews)

    for c in config:
        if c == 1:
            compute_distribution(interviews)
        elif c == 2:
            consistency_threshold = .1
            plot_type = 'spider'
            compute_consistency(interviews, plot_type, consistency_threshold)
        elif c == 3:
            compute_decisiveness(interviews)
        elif c == 4:
            model = 'rlm'
            compute_morality_correlations(interviews, model)
        elif c == 5:
            attributes = DEMOGRAPHICS
            compute_std_diff(interviews, attributes)
        elif c == 6:
            shift_threshold = 0
            attributes = DEMOGRAPHICS
            plot_morality_shifts(interviews, attributes, shift_threshold)
        elif c == 7:
            demographics_cases = [
                     (({'demographics' : {'Adolescence' : 'Late', 'Gender' : 'Male', 'Race' : 'White', 'Income' : 'Upper', 'Parent Education' : 'Tertiary'}, 'pos' : 0, 'morality' : 'Intuitive', 'ascending' : False}),
                      ({'demographics' : {'Adolescence' : 'Late', 'Gender' : 'Female', 'Race' : 'White', 'Income' : 'Upper', 'Parent Education' : 'Tertiary'}, 'pos' : 0, 'morality' : 'Intuitive', 'ascending' : False})),
                     (({'demographics' : {'Adolescence' : 'Late', 'Gender' : 'Male', 'Race' : 'White', 'Income' : 'Upper', 'Parent Education' : 'Tertiary'}, 'pos' : 0, 'morality' : 'Social', 'ascending' : True}),
                      ({'demographics' : {'Adolescence' : 'Late', 'Gender' : 'Male', 'Race' : 'White', 'Income' : 'Lower', 'Parent Education' : 'Secondary'}, 'pos' : 3, 'morality' : 'Social', 'ascending' : True}))
                    ]
            incoherent_cases = [2]
            max_diff_cases = [1]
            print_cases(interviews, demographics_cases, incoherent_cases, max_diff_cases)