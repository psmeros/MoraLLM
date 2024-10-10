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
from src.helpers import ADOLESCENCE_RANGE, CHURCH_ATTENDANCE_RANGE, CODED_WAVES, DEMOGRAPHICS, EDUCATION_RANGE, INCOME_RANGE, MORALITY_ESTIMATORS, MORALITY_ORIGIN, format_pvalue
from src.parser import prepare_data


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
    data[CODED_WAVES[0] + ':Adolescence'] = data[CODED_WAVES[0] + ':Age'].map(lambda x: ADOLESCENCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Household Income'] = data[CODED_WAVES[0] + ':Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].map(lambda x: CHURCH_ATTENDANCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Parent Education'] = data[CODED_WAVES[0] + ':Parent Education'].map(lambda x: EDUCATION_RANGE.get(x, None))

    #Prepare data
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
    # g.set(xlim=(0, 60))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_distro.png', bbox_inches='tight')
    plt.show()

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

def compute_morality_correlations(interviews, model):
    #Prepare Data
    data = interviews.copy()
    data[CODED_WAVES[1] + ':Parent Education'] = data[CODED_WAVES[0] + ':Parent Education']
    data[CODED_WAVES[1] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance']
    attribute_list = ['Verbosity', 'Uncertainty', 'Readability', 'Sentiment', 'Gender', 'Race', 'Household Income', 'Parent Education', 'Age', 'Church Attendance']
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN] + [wave + ':' + attribute for attribute in attribute_list]].values, columns=MORALITY_ORIGIN + attribute_list) for wave in CODED_WAVES]).reset_index(drop=True)

    for attribute in attribute_list[4:]:
        data[attribute] = pd.factorize(data[attribute].bfill())[0] + 1
    
    data['Household_Income'] = data['Household Income']
    data['Parent_Education'] = data['Parent Education']
    data['Church_Attendance'] = data['Church Attendance']
    data = pd.DataFrame(scale(data), columns=data.columns)

    data = data.melt(id_vars=['Verbosity', 'Uncertainty', 'Readability', 'Sentiment', 'Gender', 'Race', 'Household_Income', 'Parent_Education', 'Age', 'Church_Attendance'], value_vars=MORALITY_ORIGIN, var_name='Morality Category', value_name='morality').dropna()
    data['morality'] = data['morality'].astype(float)
    formulas = ['morality ~ Verbosity + Uncertainty + Readability + Sentiment - 1',
                'morality ~ Gender + Race + Household_Income + Parent_Education + Age + Church_Attendance - 1']

    #Display Results
    for formula in formulas:
        results = []
        for mo in MORALITY_ORIGIN:
            slice = data[data['Morality Category'] == mo]
            if model == 'ols':
                lm = smf.ols(formula=formula, data=slice).fit(cov_type='HC3')
            elif model == 'rlm':
                lm = smf.rlm(formula=formula, data=slice, M=sm.robust.norms.AndrewWave()).fit()
            results.append({param:format_pvalue((coef,pvalue)) for param, coef, pvalue in zip(lm.params.index, lm.params, lm.pvalues)})
        results = pd.DataFrame(results, index=MORALITY_ORIGIN).T
        display(results)

def compute_std_diff(interviews, attributes):
    #Prepare Data
    data = interviews.copy()
    data[CODED_WAVES[0] + ':Adolescence'] = data[CODED_WAVES[0] + ':Age'].map(lambda x: ADOLESCENCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Household Income'] = data[CODED_WAVES[0] + ':Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].map(lambda x: CHURCH_ATTENDANCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Parent Education'] = data[CODED_WAVES[0] + ':Parent Education'].map(lambda x: EDUCATION_RANGE.get(x, None))
    data = data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES] + [CODED_WAVES[0] + ':' + attribute['name'] for attribute in attributes]]

    #Melt Data
    data = data.melt(id_vars=[CODED_WAVES[0] + ':' + attribute['name'] for attribute in attributes], value_vars=[wave + ':' + mo  + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES], var_name='Morality', value_name='Value')
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

#Compute Correlations
def compute_correlations(interviews, correlation_type):
    moral_schemas = pd.concat([pd.get_dummies(interviews[CODED_WAVES[0] + ':' + 'Moral Schemas'])] * 2, ignore_index=True).astype('Int64')
    Age = pd.concat([interviews[wave + ':Age'].bfill() for wave in CODED_WAVES], ignore_index=True)
    GPA = pd.concat([interviews[CODED_WAVES[0] + ':GPA'].astype('Int64').bfill()] * 2, ignore_index=True)
    Gender = pd.Series(pd.factorize(pd.concat([interviews[wave + ':Gender'] for wave in CODED_WAVES], ignore_index=True))[0])
    Race = pd.Series(pd.factorize(pd.concat([interviews[wave + ':Race'] for wave in CODED_WAVES], ignore_index=True))[0])
    Church_Attendance = pd.concat([interviews[CODED_WAVES[0] + ':Church Attendance'].astype('Int64').bfill()] * 2, ignore_index=True)
    Parent_Education = pd.concat([interviews[CODED_WAVES[0] + ':Parent Education'].astype('Int64').bfill()] * 2, ignore_index=True)
    
    compute_pearsonr = lambda x, y: pearsonr(x, y)
    compute_fisher = lambda x, y: (lambda x, y: fisher_exact([[np.sum(x & y), np.sum(~x & y)], [np.sum(x & ~y), np.sum(~x & ~y)]]))(x.round().astype(bool), y.round().astype(bool))
    compute_rlm = lambda x, y: (lambda lm: (lm.params['x'],lm.pvalues['x']))(smf.rlm(formula='y ~ x', data=pd.concat([x, y], axis=1).rename(columns={0:'x', 1:'y'}), M=sm.robust.norms.AndrewWave()).fit())
    
    compute_correlation = lambda x, y: format_pvalue(compute_pearsonr(x, y)) if correlation_type == 'pearsonr' else format_pvalue(compute_rlm(x, y)) if correlation_type == 'rlm' else format_pvalue(compute_fisher(x, y)) if correlation_type == 'fisher' else None

    correlations = []
    for estimator in MORALITY_ESTIMATORS:
        Intuitive = pd.concat([interviews[wave + ':Intuitive_' + estimator] for wave in CODED_WAVES], ignore_index=True)
        Consequentialist = pd.concat([interviews[wave + ':Consequentialist_' + estimator] for wave in CODED_WAVES], ignore_index=True)
        Social = pd.concat([interviews[wave + ':Social_' + estimator] for wave in CODED_WAVES], ignore_index=True)
        Theistic = pd.concat([interviews[wave + ':Theistic_' + estimator] for wave in CODED_WAVES], ignore_index=True)

        correlation = {}
        correlation['Intuitive - Consequentialist'] = compute_correlation(Intuitive, Consequentialist)
        correlation['Intuitive - Social'] = compute_correlation(Intuitive, Social)
        correlation['Intuitive - Theistic'] = compute_correlation(Intuitive, Theistic)
        correlation['Consequentialist - Social'] = compute_correlation(Consequentialist, Social)
        correlation['Consequentialist - Theistic'] = compute_correlation(Consequentialist, Theistic)
        correlation['Social - Theistic'] = compute_correlation(Social, Theistic)
        
        correlation['Intuitive - Expressive Individualist'] = compute_correlation(Intuitive, moral_schemas['Expressive Individualist'])
        correlation['Intuitive - Utilitarian Individualist'] = compute_correlation(Intuitive, moral_schemas['Utilitarian Individualist'])
        correlation['Intuitive - Relational'] = compute_correlation(Intuitive, moral_schemas['Relational'])
        correlation['Intuitive - Theistic'] = compute_correlation(Intuitive, moral_schemas['Theistic'])

        correlation['Intuitive - Age'] = compute_correlation(Intuitive, Age)
        correlation['Intuitive - GPA'] = compute_correlation(Intuitive, GPA)
        correlation['Intuitive - Gender'] = compute_correlation(Intuitive, Gender)
        correlation['Intuitive - Race'] = compute_correlation(Intuitive, Race)
        correlation['Intuitive - Church Attendance'] = compute_correlation(Intuitive, Church_Attendance)
        correlation['Intuitive - Parent Education'] = compute_correlation(Intuitive, Parent_Education)
        
        correlation['Theistic - Church Attendance'] = compute_correlation(Theistic, Church_Attendance)
        correlations.append(correlation)

    correlations = pd.DataFrame(correlations, index=MORALITY_ESTIMATORS).T[MORALITY_ESTIMATORS[::-1]]
    display(correlations)

#Predict Survey Answers based on Interviews in Wave 1
def predict_behaviors(interviews, behaviors):
    for behavior in behaviors:
        #Prepare Data
        data = pd.concat([pd.DataFrame(interviews[[from_wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN] + [from_wave + ':' + c for c in behavior['Controls']] + [to_wave + ':' + a for a in behavior['Actions']]].values) for from_wave, to_wave in zip(behavior['From_Wave'], behavior['To_Wave'])])
        data.columns = MORALITY_ORIGIN + behavior['Controls'] + [a + '_pred' for a in behavior['Actions']]
        data = data.dropna()
        data[behavior['References']['Attribute Names']] = (data[behavior['References']['Attribute Names']] == behavior['References']['Attribute Values'])
        data = data.astype(float)

        #Display Results
        formulas = [a + '_pred' + ' ~ ' + ' + '.join(MORALITY_ORIGIN) + (' + ' + ' + '.join(['Q("' + c + '")' for c in behavior['Controls']]) if behavior['Controls'] else '') + ' - 1' for a in behavior['Actions']]
        results = []
        for formula in formulas:
            probit = smf.probit(formula=formula, data=data).fit(disp=False, cov_type='HC3')
            results.append({param:format_pvalue((coef,pvalue)) for param, coef, pvalue in zip(probit.params.index, probit.params, probit.pvalues)})
            
        results = pd.DataFrame(results, index=behavior['Actions']).T
        results.index = MORALITY_ORIGIN + behavior['Controls']
        display(results)

if __name__ == '__main__':
    #Hyperparameters
    config = [9]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = prepare_data(interviews)

    for c in config:
        if c == 1:
            compute_distribution(interviews)
        elif c == 2:
            consistency_threshold = .05
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
            plot_morality_distinction(interviews)
        elif c == 8:
            compute_correlations(interviews, correlation_type='pearsonr')
        elif c == 9:
            behaviors = [
                         {'From_Wave': ['Wave 1', 'Wave 3'], 
                          'To_Wave': ['Wave 2', 'Wave 4'], 
                          'Actions': ['Pot', 'Drink', 'Volunteer', 'Help'],
                          'Controls': ['Race', 'Gender', 'Age', 'Household Income', 'Parent Education', 'GPA'] + ['Pot', 'Drink', 'Volunteer', 'Help'] + ['Verbosity', 'Uncertainty', 'Readability', 'Sentiment'],
                          'References': {'Attribute Names': ['Race', 'Gender'], 'Attribute Values': ['White', 'Male']}},
                        ]
            predict_behaviors(interviews, behaviors)