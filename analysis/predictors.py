import pandas as pd
import seaborn as sns
from __init__ import *
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, zscore
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from preprocessing.constants import CODERS, MERGE_MORALITY_ORIGINS, MORALITY_ORIGIN, CODED_WAVES, MORALITY_ESTIMATORS
from preprocessing.metadata_parser import merge_codings, merge_matches, merge_surveys


def action_prediction(interviews, actions):
    #Train Classifier
    action_prediction = []
    for action in actions:
        for estimator in MORALITY_ESTIMATORS:
            input_interviews = interviews[[CODED_WAVES[0] + ':' + mo + '_' + estimator  for mo in MORALITY_ORIGIN]+[CODED_WAVES[0] + ':' + action]].dropna()
            y = input_interviews[CODED_WAVES[0] + ':' + action].apply(lambda d: False if d == 1 else True).values
            X = StandardScaler().fit_transform(input_interviews.drop([CODED_WAVES[0] + ':' + action], axis=1).values)
        
            clf = GridSearchCV(RandomForestClassifier(random_state=42), {'n_estimators': [1, 2, 3, 4, 5, 10, 20, 30], 'max_depth': [1, 2, 3, 4, 5]}, scoring='f1', cv=2)
            clf.fit(X, y)

            feature_importances = RandomForestClassifier(**clf.best_params_, random_state=42).fit(X, y).feature_importances_
            feature_importances = {item[0]:item[1] for item in sorted(zip(MORALITY_ORIGIN, feature_importances), key=lambda x: x[1], reverse=True)}
            action_prediction.append({'Action' : action, 'Estimator' : estimator, 'F1 Score' : clf.best_score_, 'Feature_Importances' : feature_importances})
    action_prediction = pd.DataFrame(action_prediction)

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(action_prediction, x='F1 Score', y='Action', hue='Estimator', hue_order=MORALITY_ESTIMATORS, orient='h', palette=sns.color_palette('Set2'))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Estimator')
    plt.savefig('data/plots/predictors-action_prediction.png', bbox_inches='tight')
    plt.show()

    action_prediction = action_prediction.groupby('Action')['Feature_Importances'].apply(list).apply(lambda l: (pd.Series(l[0]) + pd.Series(l[1])).idxmax())
    action_prediction = pd.DataFrame(action_prediction.reindex(actions).reset_index().values, columns=['Action', 'Key Morality Origin'])
    print(action_prediction)

def moral_consciousness(interviews):
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
    g = sns.lmplot(data=data[data['Estimator'] == 'Model'], x='x', y='y', row='Correlation', hue='Wave', facet_kws={'sharex':False, 'sharey':False}, seed=42, palette=sns.color_palette('Set2'))
    g.set_titles('{row_name}')
    g.set_xlabels('')
    g.set_ylabels('')
    plt.savefig('data/plots/predictors-correlations', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [1,2]
    actions=['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help']
    prefix = 'sm-' if MERGE_MORALITY_ORIGINS else ''
    interviews = pd.read_pickle('data/cache/'+prefix+'morality_model-ml-top.pkl')
    interviews = merge_surveys(interviews)
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=CODED_WAVES)

    #Filter outliers
    z_threshold = 2
    outliers = pd.DataFrame([abs(zscore(interviews[wave + ':' + morality + '_' + MORALITY_ESTIMATORS[0]])) > z_threshold for wave in CODED_WAVES for morality in ['Intuitive', 'Consequentialist', 'Social']]).any()
    interviews = interviews[~outliers]

    for c in config:
        if c == 1:
            action_prediction(interviews, actions=actions)
        elif c == 2:
            moral_consciousness(interviews)