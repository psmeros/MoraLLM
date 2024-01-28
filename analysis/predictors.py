import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer
from __init__ import *
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, zscore

from preprocessing.constants import CODERS, MERGE_MORALITY_ORIGINS, MORALITY_ORIGIN, CODED_WAVES, MORALITY_ESTIMATORS
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
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(action_prediction, x='F1-Weighted Score', y='Action', hue='Estimator', hue_order=MORALITY_ESTIMATORS, orient='h', palette=sns.color_palette('Set2'))
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
    for estimator in MORALITY_ESTIMATORS:
        correlations = pd.DataFrame(columns=CODED_WAVES, index=['Intuitive - Consequentialist', 'Social - Consequentialist', 'Intuitive - Social', 'Intuitive - Expressive Individualist', 'Intuitive - Utilitarian Individualist', 'Intuitive - Relational', 'Intuitive - Theistic', 'Intuitive - Age', 'Intuitive - GPA', 'Intuitive - Gender', 'Intuitive - Race', 'Intuitive - Church Attendance', 'Intuitive - Parent Education', 'Intuitive - Parent Income'])
        for wave in CODED_WAVES:
            if MERGE_MORALITY_ORIGINS:
                Intuitive = interviews[wave + ':Intuitive_' + estimator]
                Consequentialist = interviews[wave + ':Consequentialist_' + estimator]
                Social = interviews[wave + ':Social_' + estimator]
            else:
                Intuitive = interviews[wave + ':Experience_' + estimator]
                Consequentialist = interviews[wave + ':Consequences_' + estimator]
                Social = interviews[[wave + ':' + mo + '_' + estimator for mo in ['Family', 'Community', 'Friends']]]
                Social = Social.mean(axis=1) if estimator == 'Model' else Social.any(axis=1) if estimator == 'Coders' else None

            correlations[wave].loc['Intuitive - Consequentialist'] = compute_correlation(pearsonr(Intuitive, Consequentialist))
            correlations[wave].loc['Social - Consequentialist'] = compute_correlation(pearsonr(Social, Consequentialist))
            correlations[wave].loc['Intuitive - Social'] = compute_correlation(pearsonr(Intuitive, Social))

            data.append(pd.concat([pd.Series(Intuitive.values), pd.Series(Consequentialist.values), pd.Series(['Intuitive - Consequentialist']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))
            data.append(pd.concat([pd.Series(Social.values), pd.Series(Consequentialist.values), pd.Series(['Social - Consequentialist']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))
            data.append(pd.concat([pd.Series(Intuitive.values), pd.Series(Social.values), pd.Series(['Intuitive - Social']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))
            
            correlations[wave].loc['Intuitive - Expressive Individualist'] = compute_correlation(pearsonr(Intuitive, desicion_taking['Expressive Individualist']))
            correlations[wave].loc['Intuitive - Utilitarian Individualist'] = compute_correlation(pearsonr(Intuitive, desicion_taking['Utilitarian Individualist']))
            correlations[wave].loc['Intuitive - Relational'] = compute_correlation(pearsonr(Intuitive, desicion_taking['Relational']))
            correlations[wave].loc['Intuitive - Theistic'] = compute_correlation(pearsonr(Intuitive, desicion_taking['Theistic']))

            correlations[wave].loc['Intuitive - Age'] = compute_correlation(pearsonr(Intuitive.loc[Age.index], Age))
            correlations[wave].loc['Intuitive - GPA'] = compute_correlation(pearsonr(Intuitive.loc[Grades.index], Grades))
            correlations[wave].loc['Intuitive - Gender'] = compute_correlation(pearsonr(Intuitive, Gender))
            correlations[wave].loc['Intuitive - Race'] = compute_correlation(pearsonr(Intuitive, Race))
            correlations[wave].loc['Intuitive - Church Attendance'] = compute_correlation(pearsonr(Intuitive.loc[Church_Attendance.index], Church_Attendance))
            correlations[wave].loc['Intuitive - Parent Education'] = compute_correlation(pearsonr(Intuitive.loc[Parent_Education.index], Parent_Education))
            correlations[wave].loc['Intuitive - Parent Income'] = compute_correlation(pearsonr(Intuitive.loc[Parent_Income.index], Parent_Income))

        print(estimator)
        print(correlations)
    data = pd.concat(data, axis=0, ignore_index=True)
    data.columns = ['x', 'y', 'Correlation', 'Wave', 'Estimator']

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.lmplot(data=data[data['Estimator'] == 'Model'], x='x', y='y', col='Correlation', hue='Wave', robust=True, seed=42, palette=sns.color_palette('Set2'))
    g.set_titles('{col_name}')
    g.set_xlabels('')
    g.set_ylabels('')
    plt.savefig('data/plots/predictors-correlations', bbox_inches='tight')
    plt.show()

def compare_deviations(interviews):
    data = []
    data.append(pd.DataFrame([{'Standard Deviation' : np.std(interviews[CODED_WAVES[1] + ':Intuitive_' + estimator]) - np.std(interviews[CODED_WAVES[0] + ':Intuitive_' + estimator]), 'Morality' : 'Intuitive', 'Estimator' : estimator} for estimator in MORALITY_ESTIMATORS]))
    data.append(pd.DataFrame([{'Standard Deviation' : np.std(interviews[CODED_WAVES[1] + ':Consequentialist_' + estimator]) - np.std(interviews[CODED_WAVES[0] + ':Consequentialist_' + estimator]), 'Morality' : 'Consequentialist', 'Estimator' : estimator} for estimator in MORALITY_ESTIMATORS]))
    data.append(pd.DataFrame([{'Standard Deviation' : np.std(interviews[CODED_WAVES[1] + ':Social_' + estimator]) - np.std(interviews[CODED_WAVES[0] + ':Social_' + estimator]), 'Morality' : 'Social', 'Estimator' : estimator} for estimator in MORALITY_ESTIMATORS]))
    data = pd.concat(data)

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 10))
    ax = sns.boxplot(data, y='Estimator', x='Standard Deviation', orient='h', palette=sns.color_palette('Set2'))
    ax.set_xlabel('Standard Deviation Difference')
    plt.savefig('data/plots/predictors-deviation_comparison.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #Hyperparameters
    config = [2,3]
    actions=['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help']
    prefix = 'sm-' if MERGE_MORALITY_ORIGINS else ''
    interviews = pd.read_pickle('data/cache/'+prefix+'morality_model-top.pkl')
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