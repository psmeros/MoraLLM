import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import statsmodels.formula.api as smf
from __init__ import *
from IPython.display import display
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, distance
from scipy.stats import pearsonr, zscore
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer

from preprocessing.constants import CODED_WAVES, CODERS, MORALITY_ESTIMATORS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings, merge_matches, merge_surveys


def action_prediction(interviews, actions):
    #Train Classifier
    action_prediction = []
    for action in actions:
        for estimator in MORALITY_ESTIMATORS:
            input_interviews = interviews[[CODED_WAVES[0] + ':' + mo + '_' + estimator  for mo in MORALITY_ORIGIN]+[CODED_WAVES[0] + ':' + action]].dropna()
            y = input_interviews[CODED_WAVES[0] + ':' + action].apply(lambda d: False if d == 1 else True).values
            X = input_interviews.drop([CODED_WAVES[0] + ':' + action], axis=1).values

            classifier = LogisticRegressionCV(cv=5, random_state=42, fit_intercept=False, scoring=make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')))
            classifier.fit(X, y)
            score = np.asarray(list(classifier.scores_.values())).reshape(-1)
            coefs = {mo:coef for mo, coef in zip(MORALITY_ORIGIN, classifier.coef_[0])}
            
            action_prediction.append(pd.DataFrame({'F1-Weighted Score' : score, 'Coefs' : [coefs] * len(score), 'Action' : action, 'Estimator' : estimator}))
    action_prediction = pd.concat(action_prediction)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(action_prediction, x='F1-Weighted Score', y='Action', hue='Estimator', hue_order=MORALITY_ESTIMATORS, orient='h', palette='Set1')
    ax.set_ylabel('')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Estimator')
    plt.savefig('data/plots/predictors-action_prediction.png', bbox_inches='tight')
    plt.show()

    action_prediction = action_prediction.drop_duplicates(subset=['Action', 'Estimator', 'F1-Weighted Score']).groupby('Action')['Coefs'].apply(list).apply(lambda l: (pd.Series(l[0]) + pd.Series(l[1])).idxmax())
    action_prediction = pd.DataFrame(action_prediction.reindex(actions).reset_index().values, columns=['Action', 'Key Morality'])
    print(action_prediction)

def moral_consciousness(interviews, outlier_threshold):
    if outlier_threshold:
        outliers = pd.DataFrame([abs(zscore(interviews[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0]])) > outlier_threshold for wave in CODED_WAVES for mo in MORALITY_ORIGIN]).any()
        interviews = interviews[~outliers]
    desicion_taking = pd.get_dummies(interviews[CODED_WAVES[0] + ':' + 'Decision Taking'])
    Age = interviews[CODED_WAVES[0] + ':Age'].dropna().astype(int)
    Grades = interviews[CODED_WAVES[0] + ':Grades'].dropna().astype(int)
    Gender = pd.factorize(interviews[CODED_WAVES[0] + ':Gender'])[0]
    Race = pd.factorize(interviews[CODED_WAVES[0] + ':Race'])[0]
    Church_Attendance = interviews[CODED_WAVES[0] + ':Church Attendance'].dropna()
    Parent_Education = interviews[CODED_WAVES[0] + ':Parent Education (raw)'].dropna()
    Parent_Income = interviews[CODED_WAVES[0] + ':Income (raw)'].dropna()

    compute_correlation = lambda x: str(round(x[0], 3)).replace('0.', '.') + ('***' if float(x[1])<.005 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '')

    data = []
    correlations = pd.DataFrame(columns=[estimator + ':' + wave for estimator in MORALITY_ESTIMATORS for wave in CODED_WAVES], index=['Intuitive - Consequentialist', 'Social - Consequentialist', 'Intuitive - Social', 'Intuitive - Expressive Individualist', 'Intuitive - Utilitarian Individualist', 'Intuitive - Relational', 'Intuitive - Theistic', 'Intuitive - Age', 'Intuitive - GPA', 'Intuitive - Gender', 'Intuitive - Race', 'Intuitive - Church Attendance', 'Intuitive - Parent Education', 'Intuitive - Parent Income', 'Theistic - Church Attendance'])
    for estimator in MORALITY_ESTIMATORS:
        for wave in CODED_WAVES:
            Intuitive = interviews[wave + ':Intuitive_' + estimator]
            Consequentialist = interviews[wave + ':Consequentialist_' + estimator]
            Social = interviews[wave + ':Social_' + estimator]

            data.append(pd.concat([pd.Series(Intuitive.values), pd.Series(Consequentialist.values), pd.Series(['Intuitive - Consequentialist']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))
            data.append(pd.concat([pd.Series(Social.values), pd.Series(Consequentialist.values), pd.Series(['Social - Consequentialist']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))
            data.append(pd.concat([pd.Series(Intuitive.values), pd.Series(Social.values), pd.Series(['Intuitive - Social']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))

            correlations.loc['Intuitive - Consequentialist', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, Consequentialist))
            correlations.loc['Social - Consequentialist', estimator + ':' + wave] = compute_correlation(pearsonr(Social, Consequentialist))
            correlations.loc['Intuitive - Social', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, Social))
            
            correlations.loc['Intuitive - Expressive Individualist', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, desicion_taking['Expressive Individualist']))
            correlations.loc['Intuitive - Utilitarian Individualist', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, desicion_taking['Utilitarian Individualist']))
            correlations.loc['Intuitive - Relational', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, desicion_taking['Relational']))
            correlations.loc['Intuitive - Theistic', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, desicion_taking['Theistic']))

            correlations.loc['Intuitive - Age', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Age.index], Age))
            correlations.loc['Intuitive - GPA', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Grades.index], Grades))
            correlations.loc['Intuitive - Gender', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, Gender))
            correlations.loc['Intuitive - Race', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, Race))
            correlations.loc['Intuitive - Church Attendance', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Church_Attendance.index], Church_Attendance))
            correlations.loc['Intuitive - Parent Education', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Parent_Education.index], Parent_Education))
            correlations.loc['Intuitive - Parent Income', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Parent_Income.index], Parent_Income))
            
            correlations.loc['Theistic - Church Attendance', estimator + ':' + wave] = compute_correlation(pearsonr(interviews[wave + ':Theistic_' + estimator].loc[Church_Attendance.index], Church_Attendance))


    correlations.astype(str).to_csv('data/plots/predictors-correlations.csv')
    print(correlations)
    data = pd.concat(data, axis=0, ignore_index=True)
    data.columns = ['x', 'y', 'Correlation', 'Wave', 'Estimator']

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 20))
    g = sns.lmplot(data=data[data['Estimator'] == 'Model'], x='x', y='y', row='Correlation', hue='Wave', seed=42, palette=sns.color_palette('Set1'))
    g.set_titles('{row_name}')
    g.set_xlabels('')
    g.set_ylabels('')
    plt.savefig('data/plots/predictors-correlations', bbox_inches='tight')
    plt.show()

def compare_deviations(interviews):
    data = interviews[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]]
    stds = pd.Series([(np.std(interviews[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]]) - np.std(interviews[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]])) / np.std(interviews[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]]) for mo in MORALITY_ORIGIN], index=MORALITY_ORIGIN)
    stds = stds.apply(lambda x: str(round(x * 100, 1)) + '%')
    data = data.melt(value_vars=[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES], var_name='Morality', value_name='Value')
    data['Wave'] = data['Morality'].apply(lambda x: x.split(':')[0])
    data['Morality'] = data['Morality'].apply(lambda x: x.split(':')[1].split('_')[0])
    data['Morality'] = data['Morality'].apply(lambda x: x + ' (σ: ' + stds[x] + ')')
    data['Value'] = data['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    g = sns.displot(data, y='Wave', x='Value', col='Morality', hue='Wave', bins=20, legend=False, palette='Set1')
    g.set_titles('{col_name}')
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/predictors-deviation_comparison.png', bbox_inches='tight')
    plt.show()

def compare_areas(interviews, by_age):
    compute_convex_hull = lambda x: ConvexHull(np.diag(x).tolist()+[[0]*len(x)]).area
    if by_age:
        data = pd.concat([pd.DataFrame([interviews[wave + ':' + pd.Series(MORALITY_ORIGIN) + '_' + MORALITY_ESTIMATORS[0]].apply(lambda x: compute_convex_hull(x), axis=1).rename('Area'), interviews[wave + ':' + 'Age'].rename('Age')]).T for wave in CODED_WAVES])
        data = data.dropna()
        data['Age'] = data['Age'].astype(int)
        data['Area'] = data['Area'].astype(float)
    else:
        data = pd.DataFrame({wave : interviews[wave + ':' + pd.Series(MORALITY_ORIGIN) + '_' + MORALITY_ESTIMATORS[0]].apply(lambda x: compute_convex_hull(x), axis=1) for wave in CODED_WAVES})
        data = data.melt(value_vars=CODED_WAVES, var_name='Wave', value_name='Area')

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 5))
    if by_age:
        ax = sns.regplot(data, y='Area', x='Age')
    else:
        ax = sns.boxplot(data, y='Wave', x='Area', hue='Wave', legend=False, orient='h', palette='Set1')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Convex Hull Area')
    plt.savefig('data/plots/predictors-area_comparison.png', bbox_inches='tight')
    plt.show()

def compute_distance_distribution(interviews, include_distance):
    decisive_threshold = {wave: np.median([pd.concat([interviews[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0]].rename('Morality')]) for mo in MORALITY_ORIGIN]) for wave in CODED_WAVES}
    # decisive_threshold = {wave:.5 for wave in CODED_WAVES}

    interviews['Distance'] = interviews.apply(lambda i: [i[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]] for wave in CODED_WAVES], axis=1).apply(lambda v: distance.euclidean(v[0].to_numpy(), v[1].to_numpy()))
    morality_min_distance = interviews.loc[interviews['Distance'].sort_values(ascending=True).index[2]][[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for wave in CODED_WAVES for mo in MORALITY_ORIGIN]]
    morality_min_distance = ' -> '.join([str([round(mo, 2) for mo in morality_min_distance[:len(MORALITY_ORIGIN)]]), str([round(mo, 2) for mo in morality_min_distance[len(MORALITY_ORIGIN):]])])
    morality_min_distance_text = ' -> '.join(interviews.loc[interviews['Distance'].sort_values(ascending=True).index[2]][[wave + ':Morality_Origin'  for wave in CODED_WAVES]])
    print(morality_min_distance)
    print(morality_min_distance_text)

    morality_max_distance = interviews.loc[interviews['Distance'].sort_values(ascending=False).index[2]][[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for wave in CODED_WAVES for mo in MORALITY_ORIGIN]]
    morality_max_distance = ' -> '.join([str([round(mo, 2) for mo in morality_max_distance[:len(MORALITY_ORIGIN)]]), str([round(mo, 2) for mo in morality_max_distance[len(MORALITY_ORIGIN):]])])
    morality_max_distance_text = ' -> '.join(interviews.loc[interviews['Distance'].sort_values(ascending=False).index[2]][[wave + ':Morality_Origin'  for wave in CODED_WAVES]])
    print(morality_max_distance)
    print(morality_max_distance_text)

    #Prepare Data
    decisiveness_options = ['Decisive → Decisive', 'Indecisive → Decisive', 'Decisive → Indecisive', 'Indecisive → Indecisive']
    decisiveness = interviews.apply(lambda i: pd.Series(((i[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] > decisive_threshold[CODED_WAVES[0]]), (i[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] > decisive_threshold[CODED_WAVES[1]])) for mo in MORALITY_ORIGIN), axis=1).set_axis([mo for mo in MORALITY_ORIGIN], axis=1)
    decisiveness = decisiveness.map(lambda d: decisiveness_options[0] if d[0] and d[1] else decisiveness_options[1] if not d[0] and d[1] else decisiveness_options[2] if d[0] and not d[1] else decisiveness_options[3] if not d[0] and not d[1] else '')
    
    if include_distance:
        decisiveness = pd.concat([interviews[['Distance']], decisiveness], axis=1)
        decisiveness = decisiveness.melt(id_vars='Distance', value_vars=decisiveness.columns[1:], var_name='Morality', value_name='Decisiveness')
        decisiveness['Morality'] = decisiveness['Morality'].apply(lambda x: x.split('_')[0])

        #Plot
        sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
        plt.figure(figsize=(10, 5))
        g = sns.displot(decisiveness, x='Distance', hue='Decisiveness', col='Morality', hue_order=decisiveness_options, kind='hist', stat='probability', multiple='stack', bins=6, palette='rocket')
        g.legend.set_title('')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * len(MORALITY_ORIGIN) * 100 :.0f}%'))
        plt.savefig('data/plots/predictors-distance_distribution.png', bbox_inches='tight')
        plt.show()
    else:
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
        plt.savefig('data/plots/predictors-decisiveness.png', bbox_inches='tight')
        plt.show()



def compute_morality_wordiness_corr(interviews, wave_diff):
    #Prepare Data
    nlp = spacy.load('en_core_web_lg')
    count = lambda section : 0 if pd.isna(section) else sum([1 for token in nlp(section) if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']])
    interviews[[wave + ':Word Count' for wave in CODED_WAVES]] = interviews[[wave + ':Morality_Origin' for wave in CODED_WAVES]].map(count)

    if wave_diff:
        interviews['Word Count Diff'] = interviews[CODED_WAVES[1] + ':Word Count'] - interviews[CODED_WAVES[0] + ':Word Count']
        interviews[MORALITY_ORIGIN] = interviews[[CODED_WAVES[1] + ':' + mo for mo in MORALITY_ORIGIN]].values - interviews[[CODED_WAVES[0] + ':' + mo for mo in MORALITY_ORIGIN]].values
        interviews = interviews[MORALITY_ORIGIN + ['Word Count Diff', CODED_WAVES[0] + ':Word Count'] + [wave + ':Morality_Origin' for wave in CODED_WAVES]]

        # # #Keep values within 5th and 95th percentile
        # bounds = {mo:{'lower':interviews[mo].quantile(.05), 'upper':interviews[mo].quantile(.95)} for mo in MORALITY_ORIGIN + ['Word Count Diff', CODED_WAVES[0] + ':Word Count']}
        # interviews = interviews[pd.DataFrame([((interviews[b] >= bounds[b]['lower']) & (interviews[b] <= bounds[b]['upper'])).values for b in bounds]).all()]
        
        #Melt Data
        interviews = interviews.melt(id_vars=['Word Count Diff', CODED_WAVES[0] + ':Word Count'] + [wave + ':Morality_Origin' for wave in CODED_WAVES], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
        interviews['Value'] = interviews['Value'].astype(float)
        interviews['Word Count Diff'] = interviews['Word Count Diff'].astype(int)
        interviews[CODED_WAVES[0] + ':Word Count'] = interviews[CODED_WAVES[0] + ':Word Count'].astype(int)

        #Display Results
        results = []
        for mo in MORALITY_ORIGIN:
            data = interviews[interviews['Morality'] == mo]
            data = pd.DataFrame(data[['Value', 'Word Count Diff', CODED_WAVES[0] + ':Word Count']].values, columns=['morality', 'w31', 'w1'])
            lm = smf.ols(formula='morality ~ w31 + w1', data=data).fit()
            compute_coef = lambda x: str(round(x[0], 4)).replace('0.', '.') + ('***' if float(x[1])<.005 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '')
            results.append({param:compute_coef((coef,pvalue)) for param, coef, pvalue in zip(lm.params.index, lm.params, lm.pvalues)})
        results = pd.DataFrame(results, index=MORALITY_ORIGIN)
        display(results)
    else:
        interviews = pd.concat([pd.DataFrame(interviews[[wave + ':' + mo for mo in MORALITY_ORIGIN + ['Word Count', 'Morality_Origin', 'Wave']]].values, columns=MORALITY_ORIGIN + ['Word Count', 'Text', 'Wave']) for wave in CODED_WAVES]).reset_index()

        #Keep values within 5th and 95th percentile
        bounds = {mo:{'lower':interviews[mo].quantile(.05), 'upper':interviews[mo].quantile(.95)} for mo in MORALITY_ORIGIN + ['Word Count']}
        interviews = interviews[pd.DataFrame([((interviews[b] >= bounds[b]['lower']) & (interviews[b] <= bounds[b]['upper'])).values for b in bounds]).all()]
        
        #Melt Data
        interviews = interviews.melt(id_vars=['Word Count', 'Text', 'Wave'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
        interviews['Value'] = interviews['Value'].astype(float)
        interviews['Word Count'] = interviews['Word Count'].astype(int)

        #Compute Correlations
        compute_correlation = lambda x: str(round(x[0], 3)).replace('0.', '.') + ('***' if float(x[1])<.005 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '')
        correlations = {mo: mo + ' (r = ' + compute_correlation(pearsonr(interviews[interviews['Morality'] == mo]['Word Count'], interviews[interviews['Morality'] == mo]['Value'])) + ')' for mo in MORALITY_ORIGIN}
        interviews['Morality'] = interviews['Morality'].map(correlations)

        #Plot
        sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
        plt.figure(figsize=(10, 10))
        g = sns.lmplot(data=interviews, x='Word Count', y='Value', hue='Morality', col='Wave', seed=42, palette='Set2')
        g.set_ylabels('Morality Value')
        plt.gca().set_xlim(0)
        plt.gca().set_ylim(0,1)
        plt.savefig('data/plots/predictors-morality_wordiness_corr.png', bbox_inches='tight')
        plt.show()

def compute_morality_age_std(interviews):
    #Prepare Data
    interviews['Age'] = interviews[CODED_WAVES[0] + ':Age'].apply(lambda x: 'Early Adolescence' if x is not pd.NA and x in ['13', '14', '15'] else 'Late Adolescence' if x is not pd.NA and x in ['16', '17', '18', '19'] else '')
    interviews = interviews[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]+ ['Age']]

    #Keep values within 5th and 95th percentile
    bounds = {mo:{'lower':interviews[mo].quantile(.05), 'upper':interviews[mo].quantile(.95)} for mo in list(interviews.columns)[:-1]}
    interviews = interviews[pd.DataFrame([((interviews[b] >= bounds[b]['lower']) & (interviews[b] <= bounds[b]['upper'])).values for b in bounds]).all()]

    #Melt Data
    interviews = interviews.melt(id_vars=['Age'], value_vars=[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES], var_name='Morality', value_name='Value')
    interviews['Wave'] = interviews['Morality'].apply(lambda x: x.split(':')[0])
    interviews['Morality'] = interviews['Morality'].apply(lambda x: x.split(':')[1].split('_')[0])
    interviews['Value'] = interviews['Value'] * 100

    #Compute Standard Deviation
    stds = interviews.groupby(['Morality', 'Age', 'Wave'])['Value'].apply(np.std).reset_index()
    stds = stds.pivot(index=['Morality', 'Age'], columns='Wave', values='Value')
    stds['Value'] = (stds[CODED_WAVES[1]] - stds[CODED_WAVES[0]]) / stds[CODED_WAVES[0]]
    stds['Value'] = stds['Value'].apply(lambda x: str(round(x * 100, 1)) + '%').apply(lambda x: ' (σ: ' + ('+' if x[0] != '-' else '') + x + ')')
    stds = [age + '\n' + mo + stds.loc[mo, age]['Value']  for age in ['Early Adolescence', 'Late Adolescence'] for mo in MORALITY_ORIGIN]

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    g = sns.displot(interviews, y='Wave', x='Value', col='Morality', row='Age', row_order=['Early Adolescence', 'Late Adolescence'], hue='Wave', bins=20, legend=False, palette='Set1')
    for ax, title in zip(g.axes.flat, stds):
        ax.set_title(title)
    g.figure.subplots_adjust(hspace=0.15)
    g.figure.subplots_adjust(wspace=0.15)
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.set_xlim(0,100)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/predictors-morality_age_std.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [5]
    actions=['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help']
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews)
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=CODED_WAVES)

    for c in config:
        if c == 1:
            action_prediction(interviews, actions=actions)
        elif c == 2:
            outlier_threshold = 2
            moral_consciousness(interviews, outlier_threshold=outlier_threshold)
        elif c == 3:
            compare_deviations(interviews)
        elif c == 4:
            by_age = False
            compare_areas(interviews, by_age=by_age)
        elif c == 5:
            include_distance = False
            compute_distance_distribution(interviews, include_distance)
        elif c == 6:
            wave_diff = True
            compute_morality_wordiness_corr(interviews, wave_diff=wave_diff)
        elif c == 7:
            compute_morality_age_std(interviews)