import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from __init__ import *
from IPython.display import display

from src.helpers import CODED_WAVES, CODERS, MORALITY_ESTIMATORS, MORALITY_ORIGIN
from src.parser import merge_codings, merge_matches, merge_surveys


#Plot morality shifts
def plot_morality_shifts(interviews, attributes):

    #Compute morality shifts across waves
    def compute_morality_shifts(interviews, estimator, attribute_name=None, attribute_value=None):
        #Prepare data 
        if attribute_name is not None:
            interviews = interviews[interviews[CODED_WAVES[0] + ':' + attribute_name] == attribute_value]
        N = len(interviews)

        wave_source = interviews[[CODED_WAVES[0] + ':' + mo + '_' + estimator for mo in MORALITY_ORIGIN]]
        wave_target = interviews[[CODED_WAVES[1] + ':' + mo + '_' + estimator for mo in MORALITY_ORIGIN]]
        wave_source.columns = MORALITY_ORIGIN
        wave_target.columns = MORALITY_ORIGIN

        #Combine outgoing percentage, incoming coefficients, and remaining percentage
        outgoing = (wave_source - wave_target > 0) * abs(wave_source - wave_target)
        incoming = (wave_source - wave_target < 0) * abs(wave_source - wave_target)

        outgoing_percentage = (outgoing / wave_source).fillna(0)
        coefs = incoming.div(incoming.sum(axis=1), axis=0).fillna(0)

        remaining_percentage = (1 - outgoing_percentage)
        remaining_percentage = pd.DataFrame(np.diag(remaining_percentage.sum()), index=MORALITY_ORIGIN, columns=MORALITY_ORIGIN)

        #Normalize shift
        shift = ((outgoing_percentage.T @ coefs) + remaining_percentage) / len(interviews)

        #Reshape shift
        shift = pd.DataFrame(shift.values, index=[CODED_WAVES[0] + ':' + mo + '_' + estimator for mo in MORALITY_ORIGIN], columns=[CODED_WAVES[1] + ':' + mo + '_' + estimator for mo in MORALITY_ORIGIN])
        shift = shift.stack().reset_index().rename(columns={'level_0':'source', 'level_1':'target', 0:'value'})

        #Compute the prior shift
        shift = shift.merge(pd.DataFrame(interviews[[CODED_WAVES[0] + ':' + mo + '_' + estimator for mo in MORALITY_ORIGIN]].mean().reset_index().values, columns=['source', 'prior']))
        shift['value'] = shift['value'] * shift['prior']

        return shift, N

    #Prepare data
    shifts, _ = compute_morality_shifts(interviews, MORALITY_ESTIMATORS[0])
    shifts['wave'] = shifts.apply(lambda x: x['source'].split(':')[0] + '->' + x['target'].split(':')[0].split()[1], axis=1)
    shifts['source'] = shifts['source'].apply(lambda x: x.split(':')[-1])
    shifts['target'] = shifts['target'].apply(lambda x: x.split(':')[-1])
    source_shifts = shifts.drop('target', axis=1).rename(columns={'source':'morality'})
    source_shifts['value'] = -source_shifts['value']
    target_shifts = shifts.drop('source', axis=1).rename(columns={'target':'morality'})
    shifts = pd.concat([source_shifts, target_shifts])
    shifts['morality'] = shifts['morality'].str.split('_').apply(lambda x: x[0])
    shifts = shifts.groupby(['morality'])['value'].sum().reset_index()
    shifts['value'] = shifts['value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(20, 10))
    g = sns.catplot(data=shifts, x='value', y='morality', orient='h', order=MORALITY_ORIGIN, kind='bar', seed=42, aspect=2, color=sns.color_palette('Set1')[2])
    g.set(xlim=(-7, 7))
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.set_ylabel('')
    plt.title('Overall Shift')
    plt.savefig('data/plots/substantive-morality_shift.png', bbox_inches='tight')
    plt.show()

    #Prepare data
    shifts = []
    for estimator in MORALITY_ESTIMATORS:
        for i, attribute in enumerate(attributes):
            attributes[i]['N'] = {}
            for j, attribute_value in enumerate(attribute['values']):
                shift, N = compute_morality_shifts(interviews, estimator=estimator, attribute_name=attribute['name'], attribute_value=attribute_value)
                if not shift.empty:
                    shift['Attribute'] = attribute['name']
                    attributes[i]['N'][j] = attribute_value #+ ' (N = ' + str(N) + ')'
                    shift['Attribute Position'] = j
                    shift['Estimator'] = estimator
                    shifts.append(shift)
    shifts = pd.concat(shifts)
    shifts['wave'] = shifts.apply(lambda x: x['source'].split(':')[0] + '->' + x['target'].split(':')[0].split()[1], axis=1)
    shifts['source'] = shifts['source'].apply(lambda x: x.split(':')[-1])
    shifts['target'] = shifts['target'].apply(lambda x: x.split(':')[-1])
    source_shifts = shifts.drop('target', axis=1).rename(columns={'source':'morality'})
    source_shifts['value'] = -source_shifts['value']
    target_shifts = shifts.drop('source', axis=1).rename(columns={'target':'morality'})
    shifts = pd.concat([source_shifts, target_shifts])
    shifts['morality'] = shifts['morality'].str.split('_').apply(lambda x: x[0])
    shifts = shifts.groupby(['morality', 'Estimator', 'Attribute', 'Attribute Position'])['value'].sum().reset_index()
    shifts['value'] = shifts['value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(20, 10))
    g = sns.catplot(data=shifts[shifts['Estimator'] == MORALITY_ESTIMATORS[0]], x='value', y='morality', hue='Attribute Position', orient='h', order=MORALITY_ORIGIN, col='Attribute', col_order=[attribute['name'] for attribute in attributes], col_wrap=3, kind='bar', legend=False, seed=42, palette='Set1')
    g.set(xlim=(-7, 7))
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    g.set_titles('')
    for (j, attribute), pos in zip(enumerate(g.col_names), [(ax.get_position().x0 + ax.get_position().x1)/2 - (i%3)*.03 for i, ax in enumerate(g.axes)]):
        g = g.add_legend(title=attribute, legend_data={v:mpatches.Patch(color=sns.color_palette('Set1')[i]) for i, v in enumerate(attributes[j]['N'].values())}, bbox_to_anchor=(pos - (len(''.join(attributes[0]['N'].values()))/2)*0.001, 1.1 - (j // 3) * 0.5), adjust_subtitles=True, loc='upper center')
    for ax in g.axes:
        ax.set_ylabel('')
    plt.subplots_adjust(hspace=0.6)
    plt.savefig('data/plots/substantive-morality_shift_by_attribute.png', bbox_inches='tight')
    plt.show()

def compute_decisiveness(interviews):
    decisive_threshold = {wave: np.median([pd.concat([interviews[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0]].rename('Morality')]) for mo in MORALITY_ORIGIN]) for wave in CODED_WAVES}
    # decisive_threshold = {wave:.5 for wave in CODED_WAVES}

    #Prepare Data
    decisiveness_options = ['Decisive → Decisive', 'Indecisive → Decisive', 'Decisive → Indecisive', 'Indecisive → Indecisive']
    decisiveness = interviews.apply(lambda i: pd.Series(((i[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] > decisive_threshold[CODED_WAVES[0]]), (i[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] > decisive_threshold[CODED_WAVES[1]])) for mo in MORALITY_ORIGIN), axis=1).set_axis([mo for mo in MORALITY_ORIGIN], axis=1)
    decisiveness = decisiveness.map(lambda d: decisiveness_options[0] if d[0] and d[1] else decisiveness_options[1] if not d[0] and d[1] else decisiveness_options[2] if d[0] and not d[1] else decisiveness_options[3] if not d[0] and not d[1] else '')
    
    decisiveness = decisiveness.apply(lambda x: x.value_counts(normalize=True) * 100).T
    decisiveness = decisiveness[decisiveness_options]

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    decisiveness.plot(kind='barh', stacked=True, color=pd.Series(sns.color_palette('vlag'))[[0,1,4,5]])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y :.0f}%'))
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(1, 1.03)).set_frame_on(False)
    plt.savefig('data/plots/substantive-decisiveness.png', bbox_inches='tight')
    plt.show()

def compute_morality_wordiness_corr(interviews):
    #Prepare Data
    interviews['Word Count Diff'] = interviews[CODED_WAVES[1] + ':Morality_Origin_Word_Count'] - interviews[CODED_WAVES[0] + ':Morality_Origin_Word_Count']
    interviews[MORALITY_ORIGIN] = interviews[[CODED_WAVES[1] + ':' + mo for mo in MORALITY_ORIGIN]].values - interviews[[CODED_WAVES[0] + ':' + mo for mo in MORALITY_ORIGIN]].values
    interviews = interviews[MORALITY_ORIGIN + ['Word Count Diff', CODED_WAVES[0] + ':Morality_Origin_Word_Count'] + [wave + ':Morality_Origin' for wave in CODED_WAVES]]

    #Melt Data
    interviews = interviews.melt(id_vars=['Word Count Diff', CODED_WAVES[0] + ':Morality_Origin_Word_Count'] + [wave + ':Morality_Origin' for wave in CODED_WAVES], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    interviews['Value'] = interviews['Value'].astype(float)
    interviews['Word Count Diff'] = interviews['Word Count Diff'].astype(int)
    interviews[CODED_WAVES[0] + ':Morality_Origin_Word_Count'] = interviews[CODED_WAVES[0] + ':Morality_Origin_Word_Count'].astype(int)

    #Display Results
    results = []
    for mo in MORALITY_ORIGIN:
        data = interviews[interviews['Morality'] == mo]
        data = pd.DataFrame(data[['Value', 'Word Count Diff', CODED_WAVES[0] + ':Morality_Origin_Word_Count']].values, columns=['morality', 'w31', 'w1'])
        lm = smf.ols(formula='morality ~ w31', data=data).fit()
        compute_coef = lambda x: str(round(x[0], 4)).replace('0.', '.') + ('***' if float(x[1])<.005 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '')
        results.append({param:compute_coef((coef,pvalue)) for param, coef, pvalue in zip(lm.params.index, lm.params, lm.pvalues)})
    results = pd.DataFrame(results, index=MORALITY_ORIGIN)
    display(results)

    #Keep values within 5th and 95th percentile
    # bounds = {mo:{'lower':interviews[mo].quantile(.05), 'upper':interviews[mo].quantile(.95)} for mo in ['Word Count Diff']}
    # interviews = interviews[pd.DataFrame([((interviews[b] >= bounds[b]['lower']) & (interviews[b] <= bounds[b]['upper'])).values for b in bounds]).all()]

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.lmplot(data=interviews, x='Word Count Diff', y='Value', hue='Morality', seed=42, palette='Set2')
    g.set_ylabels('Morality Value Diff')
    plt.gca().set_ylim(-1,1)
    plt.savefig('data/plots/substantive-morality_wordiness_corr.png', bbox_inches='tight')
    plt.show()

def compute_morality_age_corr(interviews):
    #Prepare Data
    data = pd.concat([pd.DataFrame(interviews[[wave + ':' + mo for mo in MORALITY_ORIGIN + ['Age']]].values, columns=MORALITY_ORIGIN+['Age']) for wave in CODED_WAVES]).dropna().reset_index(drop=True)
    data['Age'] = data['Age'].astype(int)
    data['Average'] = data[MORALITY_ORIGIN].mean(axis=1)

    #Melt Data
    data = data.melt(id_vars=['Age'], value_vars=MORALITY_ORIGIN+['Average'], var_name='Morality', value_name='Value').dropna()
    data['Value'] = data['Value'].astype(float)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.lmplot(data=data, x='Age', y='Value', hue='Morality', scatter=False, seed=42, robust=True, aspect=1.2, palette=sns.color_palette('Set2'))
    g.set_titles('{row_name}')
    g.set_ylabels('Morality Value')
    plt.savefig('data/plots/substantive-morality_age_lm', bbox_inches='tight')
    plt.show()

    #Prepare Data
    interviews['Age'] = interviews[CODED_WAVES[0] + ':Age'].apply(lambda x: 'Early Adolescence' if x is not pd.NA and x in ['13', '14', '15'] else 'Late Adolescence' if x is not pd.NA and x in ['16', '17', '18', '19'] else '')
    data = interviews[[wave + ':' + mo for mo in MORALITY_ORIGIN for wave in CODED_WAVES]+ ['Age']]

    #Keep values within 5th and 95th percentile
    # bounds = {mo:{'lower':data[mo].quantile(.05), 'upper':data[mo].quantile(.95)} for mo in list(data.columns)[:-1]}
    # data = data[pd.DataFrame([((data[b] >= bounds[b]['lower']) & (data[b] <= bounds[b]['upper'])).values for b in bounds]).all()]

    #Melt Data
    data = data.melt(id_vars=['Age'], value_vars=[wave + ':' + mo for mo in MORALITY_ORIGIN for wave in CODED_WAVES], var_name='Morality', value_name='Value')
    data['Wave'] = data['Morality'].apply(lambda x: x.split(':')[0])
    data['Morality'] = data['Morality'].apply(lambda x: x.split(':')[1].split('_')[0])
    data['Value'] = data['Value'] * 100

    #Compute Standard Deviation
    stds = data.groupby(['Morality', 'Age', 'Wave'])['Value'].apply(np.std).reset_index()
    stds = stds.pivot(index=['Morality', 'Age'], columns='Wave', values='Value')
    stds['Value'] = (stds[CODED_WAVES[1]] - stds[CODED_WAVES[0]]) / stds[CODED_WAVES[0]]
    stds['Value'] = stds['Value'].apply(lambda x: str(round(x * 100, 1)) + '%').apply(lambda x: ' (σ: ' + ('+' if x[0] != '-' else '') + x + ')')
    stds = [age + '\n' + mo + stds.loc[mo, age]['Value']  for age in ['Early Adolescence', 'Late Adolescence'] for mo in MORALITY_ORIGIN]

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    g = sns.displot(data, y='Wave', x='Value', col='Morality', row='Age', row_order=['Early Adolescence', 'Late Adolescence'], hue='Wave', bins=20, legend=False, palette='Set1')
    for ax, title in zip(g.axes.flat, stds):
        ax.set_title(title)
    g.figure.subplots_adjust(hspace=0.15)
    g.figure.subplots_adjust(wspace=0.15)
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.set_xlim(0,100)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/substantive-morality_age_std.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [1,2,3,4]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')

    # #Keep values within 5th and 95th percentile
    # bounds = {mo:{'lower':interviews[mo].quantile(.05), 'upper':interviews[mo].quantile(.95)} for mo in MORALITY_ORIGIN}
    # interviews = interviews[pd.DataFrame([((interviews[b] >= bounds[b]['lower']) & (interviews[b] <= bounds[b]['upper'])).values for b in bounds]).all()]

    interviews = merge_surveys(interviews)
    interviews = merge_codings(interviews)
    interviews = merge_matches(interviews)

    for c in config:
        if c == 1:
            compute_decisiveness(interviews)
        elif c == 2:
            compute_morality_wordiness_corr(interviews)
        elif c == 3:
            compute_morality_age_corr(interviews)
        elif c == 4:
            interviews[CODED_WAVES[0] + ':Race'] = interviews[CODED_WAVES[0] + ':Race'].apply(lambda x: x if x in ['White'] else 'Other')
            interviews[CODED_WAVES[0] + ':Age'] = interviews[CODED_WAVES[0] + ':Age'].apply(lambda x: 'Early Adolescence' if x is not pd.NA and x in ['13', '14', '15'] else 'Late Adolescence' if x is not pd.NA and x in ['16', '17', '18', '19'] else '')
            interviews[CODED_WAVES[0] + ':Church Attendance'] = interviews[CODED_WAVES[0] + ':Church Attendance'].apply(lambda x: 'Irregular' if x is not pd.NA and x in [1,2,3,4] else 'Regular' if x is not pd.NA and x in [5,6] else '')
            attributes = [{'name' : 'Gender', 'values' : ['Male', 'Female']},
                        {'name' : 'Race', 'values' : ['White', 'Other']},
                        {'name' : 'Income', 'values' : ['Upper', 'Lower']},
                        {'name' : 'Parent Education', 'values' : ['Tertiary', 'Secondary']},
                        {'name' : 'Age', 'values' : ['Early Adolescence', 'Late Adolescence']},
                        {'name' : 'Church Attendance', 'values' : ['Regular', 'Irregular']}]
            plot_morality_shifts(interviews, attributes)